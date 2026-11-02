"""CPU learning runtime: evidence is durable; interventions become real on commit."""

import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from jsonschema import ValidationError

from apps.learning_service import schemas
from apps.learning_service.store import Store, canonical
from packages.learning.actions import (
    decode_navigator,
    decode_strategist,
    encode_strategist,
    navigator_mask,
    strategist_mask,
)
from packages.learning.belief import ConceptBelief, update_belief
from packages.learning.content.templates.families import families_for
from packages.learning.episode import (
    _active_context,
    _build_plan,
    _navigator_graph,
    _navigator_observation,
    _next_item,
    _public_state,
    load_graph,
    update_public_misconceptions,
)
from packages.learning.features import build_features
from packages.learning.grader import grade_answer
from packages.learning.scheduler import GoalStack, boundary_reasons
from packages.policies.rules import RulesNavigator, RulesStrategist
from packages.simulator.randomness import EventRandomBank

HTTP_STATUS = dict(
    INVALID_SCHEMA=422,
    UNSUPPORTED_VERSION=400,
    UNAUTHORIZED=401,
    FORBIDDEN=403,
    STATE_CONFLICT=409,
    PLAN_EXPIRED=409,
    SESSION_ENDED=409,
    RATE_LIMITED=429,
    GENERATION_UNAVAILABLE=503,
    LEARNING_UNAVAILABLE=503,
    DEADLINE_EXCEEDED=504,
)


class ServiceError(Exception):
    def __init__(self, code, message, *, retryable=False, current_state_version=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.current_state_version = current_state_version

    @property
    def http_status(self):
        return HTTP_STATUS[self.code]

    def as_dict(self, request_id, trace_id):
        return error_body(self, request_id, trace_id)


def error_body(error, request_id, trace_id):
    body = dict(
        schema_version="1.0",
        request_id=request_id,
        trace_id=trace_id,
        code=error.code,
        message=error.message,
        retryable=error.retryable,
        current_state_version=error.current_state_version,
    )
    schemas.validate("Error", body)
    return body


def build_strategist(config):
    selection = config.get("strategist", "rules")
    if selection == "rules":
        return RulesStrategist()
    if isinstance(selection, dict) and set(selection) == {"checkpoint"}:
        from packages.policies.network import NetworkStrategist

        return NetworkStrategist(selection["checkpoint"], greedy=True)
    raise ValueError("Unsupported strategist configuration")


def _hash(body):
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def _request_id(body):
    return body.get("request_id") if isinstance(body, dict) else None


def _validate(name, body):
    if isinstance(body, dict):
        version = body.get("schema_version")
        if isinstance(version, str) and version.split(".")[0] != "1":
            raise ServiceError("UNSUPPORTED_VERSION", "Unsupported schema major version")
        evidence = body.get("evidence")
        if isinstance(evidence, dict):
            _validate("Evidence", evidence)
    try:
        schemas.validate(name, body)
    except (ValidationError, ValueError, TypeError):
        raise ServiceError("INVALID_SCHEMA", "Request does not match the schema") from None


class LearningEngine:
    def __init__(self, config, db_path, *, clock=None, strategist=None):
        self.config = dict(config)
        self.store = Store(db_path)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.strategist = strategist if strategist is not None else build_strategist(config)
        self.navigator = RulesNavigator()
        self.graph = load_graph()
        self.graph["enabled_concepts"] = list(
            config.get("enabled_concepts", self.graph["enabled_concepts"])
        )
        self.budget = config.get("teaching_budget", 16)
        if not isinstance(self.budget, int) or not 0 <= self.budget <= 16:
            raise ValueError("Teaching budget must be between zero and sixteen")

    def close(self):
        self.store.close()

    def _now(self):
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() != timedelta(0):
            raise ValueError("Clock must return an aware UTC datetime")
        return now

    def _event(self, db, sid, kind, payload):
        db.execute(
            "INSERT INTO events SELECT ?, COALESCE(MAX(seq), 0)+1, ?, ?, ? "
            "FROM events WHERE session_id=?",
            (sid, kind, canonical(payload), self._now().isoformat(), sid),
        )

    def _save(self, db, state):
        db.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?)", (state["session_id"], canonical(state))
        )

    def _session(self, db, sid, *, active=True):
        row = db.execute("SELECT state_json FROM sessions WHERE session_id=?", (sid,)).fetchone()
        if row is None:
            raise ServiceError("INVALID_SCHEMA", "Unknown session")
        state = json.loads(row[0])
        if active and state["status"] == "ended":
            self._fail("SESSION_ENDED", "Session has ended", state)
        return state

    @staticmethod
    def _fail(code, message, state):
        raise ServiceError(code, message, current_state_version=state["state_version"])

    def _plan(self, db, pid):
        row = db.execute("SELECT * FROM plans WHERE plan_id=?", (pid,)).fetchone()
        if row is None:
            raise ServiceError("INVALID_SCHEMA", "Unknown plan")
        return dict(row)

    def _expire(self, db, row):
        if row["status"] == "pending" and self._now() >= datetime.fromisoformat(row["expires_at"]):
            db.execute(
                "UPDATE plans SET status='canceled', canceled_reason='expired' WHERE plan_id=?",
                (row["plan_id"],),
            )
            row.update(status="canceled", canceled_reason="expired")
            self._event(db, row["session_id"], "plan_expired", {"plan_id": row["plan_id"]})

    def _post(self, endpoint, body, key, request_schema, response_schema, operation):
        _validate(request_schema, body)
        _validate("IdempotencyKey", key)
        digest = _hash(body)
        failure = None
        with self.store.transaction() as db:
            cached = db.execute(
                "SELECT * FROM idempotency WHERE endpoint=? AND key=?", (endpoint, key)
            ).fetchone()
            if cached:
                if cached["body_hash"] != digest:
                    version = None
                    if "session_id" in body:
                        session = db.execute(
                            "SELECT state_json FROM sessions WHERE session_id=?",
                            (body["session_id"],),
                        ).fetchone()
                        if session:
                            version = json.loads(session[0])["state_version"]
                    raise ServiceError(
                        "STATE_CONFLICT",
                        "Idempotency key reused with different body",
                        current_state_version=version,
                    )
                return json.loads(cached["response_json"])
            result = operation(db)
            if isinstance(result, ServiceError):
                failure = result
            else:
                schemas.validate(response_schema, result)
                db.execute(
                    "INSERT INTO idempotency VALUES (?, ?, ?, ?)",
                    (endpoint, key, digest, canonical(result)),
                )
        if failure:
            raise failure
        return result

    @staticmethod
    def _public(state):
        fields = (
            "session_id",
            "state_version",
            "current_concept_id",
            "goal_concept_id",
            "teaching_turns",
            "status",
            "end_reason",
        )
        return {key: state[key] for key in fields}

    def create_session(self, body, *, trace_id):
        def create(db):
            if body["goal_concept_id"] not in self.graph["enabled_concepts"]:
                raise ServiceError("INVALID_SCHEMA", "Goal concept is not enabled")
            state = dict(
                session_id=f"s-{uuid4().hex}",
                state_version=0,
                status="active",
                end_reason=None,
                current_concept_id=body["goal_concept_id"],
                goal_concept_id=body["goal_concept_id"],
                pace=body["pace_preference"],
                teaching_turns=0,
                teaching_budget=self.budget,
                beliefs={
                    f"C{i:02d}": {**asdict(ConceptBelief()), "recent_independent": []}
                    for i in range(1, 16)
                },
                stack=[],
                visit_index=0,
                last_pair_visit=[],
                pair_visits=[],
                option_turns=0,
                success_streak=0,
                failure_streak=0,
                hint_count=0,
                meaningful_error=False,
                unsuccessful_hints=0,
                no_diagnostic_yet=True,
                skip_deadline=None,
                skip_diagnostic_done=False,
                active_item=None,
                previous_move="NONE",
                used_assessment={},
                assessment_family_reuse_counts={},
                used_teaching=[],
                last_response_id=None,
                last_plan=None,
                applied_evidence=[],
                navigation_action_id=None,
                plan_counter=0,
                requests=dict(
                    skip=False,
                    faster=False,
                    slower=False,
                    stop=False,
                    answer=False,
                    goal=None,
                    goal_change=False,
                ),
                public_misconceptions={},
            )
            self._save(db, state)
            self._event(db, state["session_id"], "session_created", self._public(state))
            return dict(
                schema_version="1.0",
                session_id=state["session_id"],
                public_state=self._public(state),
            )

        return self._post(
            "create",
            body,
            _request_id(body),
            "CreateSessionRequest",
            "CreateSessionResponse",
            create,
        )

    @staticmethod
    def _beliefs(state):
        return {
            cid: ConceptBelief(
                **{
                    **data,
                    "recent_independent": deque(
                        (tuple(item) for item in data["recent_independent"]), maxlen=5
                    ),
                }
            )
            for cid, data in state["beliefs"].items()
        }

    @staticmethod
    def _pack_beliefs(state, beliefs):
        state["beliefs"] = {
            cid: {**asdict(b), "recent_independent": list(b.recent_independent)}
            for cid, b in beliefs.items()
        }

    @staticmethod
    def _item(data):
        if data is None:
            return None
        family = next(
            f
            for f in families_for(data["concept_id"], role=data["role"], split=data["split"])
            if f.template_family_id == data["family_id"]
        )
        return dict(
            family=family,
            instance=family.sample(data["seed"]),
            item_id=data["item_id"],
            hint_level=data["hint_level"],
        )

    def _apply_evidence(self, db, state, evidence):
        if state["last_response_id"] is None:
            if evidence is not None:
                self._fail("INVALID_SCHEMA", "First observation requires null evidence", state)
            return
        if (
            evidence is None
            or evidence["session_id"] != state["session_id"]
            or (evidence["source_response_id"] != state["last_response_id"])
        ):
            self._fail(
                "INVALID_SCHEMA", "Evidence must reference the last committed response", state
            )
        if evidence["problem_instance_id"] != state["last_plan"]["problem_instance_id"]:
            self._fail("INVALID_SCHEMA", "Evidence problem reference does not match", state)
        if evidence["event_id"] in state["applied_evidence"]:
            return
        target = evidence["requested_concept_id"]
        request = evidence["explicit_request"]
        if target is not None and target not in self.graph["enabled_concepts"]:
            self._fail("INVALID_SCHEMA", "Requested concept is not enabled", state)
        if request == "change_goal" and target is None:
            self._fail("INVALID_SCHEMA", "Goal change requires an enabled concept", state)
        item = self._item(state["active_item"])
        plan = state["last_plan"]
        correctness = "unknown"
        if item and plan["problem_instance_id"] == item["item_id"]:
            if evidence["normalized_answer"] is not None:
                correctness = grade_answer(
                    evidence["normalized_answer"], item["instance"], item["family"]
                )
            state["public_misconceptions"] = update_public_misconceptions(
                state.get("public_misconceptions", {}),
                family=item["family"],
                instance=item["instance"],
                normalized_answer=evidence["normalized_answer"],
                correctness=correctness,
                move=plan["teaching_action"]["move"],
            )
            assistance = evidence["assistance"]
            if plan["assistance"] != "none":
                assistance = plan["assistance"]
            beliefs = self._beliefs(state)
            update_belief(
                beliefs[plan["concept_id"]],
                correctness,
                assistance=assistance,
                move=plan["teaching_action"]["move"],
                family_id=plan["template_family_id"],
                event_id=evidence["event_id"],
            )
            self._pack_beliefs(state, beliefs)
            if assistance == "none" and correctness in {"correct", "incorrect"}:
                correct = correctness == "correct"
                state["success_streak"] = state["success_streak"] + 1 if correct else 0
                state["failure_streak"] = state["failure_streak"] + 1 if not correct else 0
                state["meaningful_error"] = not correct
            elif assistance != "none" and correctness in {"correct", "incorrect"}:
                state["unsuccessful_hints"] += 1
            if plan["teaching_action"]["move"] == "HINT" and correctness == "correct":
                state["unsuccessful_hints"] = 0
        requests = state["requests"]
        if request in {"skip", "faster", "slower", "stop", "answer"}:
            requests[request] = True
        if request == "hint":
            state["meaningful_error"] = True
        if request == "skip":
            requests["goal"] = target
            if target is not None:
                state["goal_concept_id"] = target
        if request in {"faster", "slower"}:
            state["pace"] = "concise" if request == "faster" else "detailed"
        if request == "change_goal":
            state["goal_concept_id"] = target
            requests.update(goal=target, goal_change=True)
        state["applied_evidence"].append(evidence["event_id"])
        self._event(db, state["session_id"], "evidence_applied", evidence)

    def _context(self, state, beliefs, item, stack):
        nav = state["navigation_action_id"]
        operation, index = decode_navigator(nav) if nav is not None else (None, None)
        public = _public_state(
            beliefs,
            current=state["current_concept_id"],
            goal=state["goal_concept_id"],
            turn=state["teaching_turns"],
            active_item=item,
            active_option_type=operation,
            active_target=f"C{index + 1:02d}" if index is not None else None,
            previous_move=state["previous_move"],
            option_turns=state["option_turns"],
            success_streak=state["success_streak"],
            failure_streak=state["failure_streak"],
            hint_count=state["hint_count"],
            explicit_skip=state["requests"]["skip"],
            explicit_faster=state["requests"]["faster"],
            explicit_slower=state["requests"]["slower"],
            stack=stack.stack,
            misconceptions=state.get("public_misconceptions", {}),
        )
        public["remaining_budget"] = state["teaching_budget"] - state["teaching_turns"]
        context = _active_context(
            public,
            current=state["current_concept_id"],
            goal=state["goal_concept_id"],
            beliefs=beliefs,
            active_item=item,
            stack=stack,
            graph=self.graph,
            requests=state["requests"],
            success_streak=state["success_streak"],
            failure_streak=state["failure_streak"],
            meaningful_error=state["meaningful_error"],
            unsuccessful_hints=state["unsuccessful_hints"],
            no_diagnostic_yet=state["no_diagnostic_yet"],
            skip_deadline=state["skip_deadline"],
            skip_diagnostic_done=state["skip_diagnostic_done"],
            turn=state["teaching_turns"],
            pace=state["pace"],
            allow_policy_stop=False,
        )
        return public, context

    def _end(self, db, state, reason):
        state.update(status="ended", end_reason=reason, state_version=state["state_version"] + 1)
        self._save(db, state)
        result = dict(
            schema_version="1.0",
            session_id=state["session_id"],
            state_version=state["state_version"],
            reason=reason,
            summary_content_id=f"summary-{reason}",
            public_state=self._public(state),
        )
        self._event(db, state["session_id"], "session_ended", result)
        return dict(kind="session_end", session_end=result)

    def observe_and_plan(self, body, *, trace_id):
        def observe(db):
            state = self._session(db, body["session_id"])
            if body["expected_state_version"] != state["state_version"]:
                self._fail("STATE_CONFLICT", "State version is stale", state)
            pending = db.execute(
                "SELECT plan_id FROM plans WHERE session_id=? AND status='pending'",
                (state["session_id"],),
            ).fetchone()
            if pending:
                row = self._plan(db, pending[0])
                self._expire(db, row)
                if row["status"] == "pending":
                    self._fail("STATE_CONFLICT", "A plan is already pending", state)
            self._apply_evidence(db, state, body["evidence"])
            beliefs = self._beliefs(state)
            if state["requests"]["stop"]:
                return self._end(db, state, "learner_stop")
            if beliefs[state["goal_concept_id"]].certified:
                return self._end(db, state, "demonstrated_completion")
            if state["teaching_turns"] >= state["teaching_budget"]:
                return self._end(db, state, "teaching_budget")
            state["plan_counter"] += 1
            self._save(db, state)
            proposed = copy.deepcopy(state)
            try:
                plan, item_data = self._propose(proposed, trace_id)
            except ServiceError as error:
                return error
            if plan is None:
                return self._end(db, state, "policy_stop")
            db.execute(
                "INSERT INTO plans (plan_id, session_id, status, plan_json, state_json, item_json, "
                "reserved_at, expires_at) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)",
                (
                    plan["plan_id"],
                    state["session_id"],
                    canonical(plan),
                    canonical(proposed),
                    canonical(item_data),
                    self._now().isoformat(),
                    plan["expires_at"],
                ),
            )
            self._event(db, state["session_id"], "plan_reserved", {"plan_id": plan["plan_id"]})
            return dict(kind="plan", teaching_plan=plan)

        return self._post(
            "observe", body, _request_id(body), "ObserveRequest", "ObserveResponse", observe
        )

    def _propose(self, state, trace_id):
        beliefs = self._beliefs(state)
        stack = GoalStack(
            stack=state["stack"],
            visit_index=state["visit_index"],
            last_pair_visit={tuple(k): v for k, v in state["last_pair_visit"]},
            pair_visits=defaultdict(int, {tuple(k): v for k, v in state["pair_visits"]}),
        )
        item = self._item(state["active_item"])
        public, context = self._context(state, beliefs, item, stack)
        reasons = boundary_reasons(
            {
                **context,
                "session_start": state["navigation_action_id"] is None,
                "option_turns": state["option_turns"],
                "content_family_exhausted": bool(
                    state["previous_move"] in {"PRACTICE", "CHALLENGE"}
                    and {
                        f.template_family_id
                        for f in families_for(
                            state["current_concept_id"], role="teaching", split="train"
                        )
                    }
                    <= set(state["used_teaching"])
                ),
            }
        )
        if reasons:
            if "family_exhausted" in reasons:
                state["used_teaching"] = []
            mask = navigator_mask(
                context,
                _navigator_graph(
                    self.graph, state["goal_concept_id"], state["current_concept_id"], beliefs
                ),
            )
            action = self.navigator.act(_navigator_observation(public), mask, context)
            operation, index = decode_navigator(action)
            target = f"C{index + 1:02d}" if index is not None else None
            if operation == "END_SESSION":
                return None, None
            old_current = state["current_concept_id"]
            if operation == "REVIEW_PREREQUISITE" and target:
                if stack.push(state["goal_concept_id"], old_current):
                    state["current_concept_id"] = target
            elif operation == "RETURN_TO_GOAL" and target:
                stack.pop()
                state["current_concept_id"] = target
            elif operation in {"SKIP_TO_TARGET", "ADVANCE_ONE"} and target:
                state["current_concept_id"] = target
                if operation == "SKIP_TO_TARGET":
                    state.update(
                        skip_deadline=state["teaching_turns"] + 2, skip_diagnostic_done=False
                    )
            state["navigation_action_id"] = action
            state["option_turns"] = 0
            state["success_streak"] = state["failure_streak"] = 0
            state["requests"].update(skip=False, goal_change=False)
            if old_current != state["current_concept_id"]:
                state.update(active_item=None, no_diagnostic_yet=True)
                item = None
            public, context = self._context(state, beliefs, item, stack)
        mask = strategist_mask(context)
        action = self.strategist.act(build_features(public), mask, context)
        if not isinstance(action, int) or not 0 <= action < len(mask) or not mask[action]:
            raise ServiceError(
                "LEARNING_UNAVAILABLE",
                "Policy returned an invalid action",
                retryable=True,
                current_state_version=state["state_version"],
            )
        forced = state["requests"]["answer"] and not (
            state["last_plan"] and state["last_plan"]["assessment_mode"] == "independent"
        )
        if forced:
            _, difficulty, pace = decode_strategist(action)
            action = encode_strategist("EXPLAIN_CONCEPT", difficulty, pace)
        move, _, _ = decode_strategist(action)
        sid = state["session_id"]
        count = state["plan_counter"]
        scenario = dict(
            scenario_id=f"{sid}-{count}",
            split="train",
            template_family_allowlist=[
                f.template_family_id
                for cid in self.graph["enabled_concepts"]
                for role in ("teaching", "assessment")
                for f in families_for(cid, role=role, split="train")
            ],
        )
        if move in {"PROBE", "PRACTICE", "CHALLENGE", "CHECK_MASTERY"}:
            bank = EventRandomBank()
            seed = self.config.get("seed", 0)
            assessment_family_uses, used_teaching = (
                Counter(state["used_assessment"]),
                set(state["used_teaching"]),
            )
            assessment_family_reuse_counts = Counter(
                state.get("assessment_family_reuse_counts", {})
            )
            family, instance, item_id = _next_item(
                state["current_concept_id"],
                move,
                scenario,
                seed,
                count,
                bank,
                assessment_family_uses=assessment_family_uses,
                assessment_family_reuse_counts=assessment_family_reuse_counts,
                used_teaching_families=used_teaching,
            )
            sample_seed = int(
                bank.rng(scenario["scenario_id"], seed, count, "item_parameters").integers(0, 2**31)
            )
            state["active_item"] = dict(
                concept_id=state["current_concept_id"],
                role=family.role,
                split="train",
                family_id=family.template_family_id,
                seed=sample_seed,
                item_id=item_id,
                hint_level=0,
            )
            item = dict(family=family, instance=instance, item_id=item_id, hint_level=0)
            if family.role == "teaching":
                used_teaching.add(family.template_family_id)
            state.update(
                used_assessment=dict(assessment_family_uses),
                assessment_family_reuse_counts=dict(assessment_family_reuse_counts),
                used_teaching=sorted(used_teaching),
            )
        elif move == "HINT" and item:
            item["hint_level"] = min(3, item["hint_level"] + 1)
            state["active_item"]["hint_level"] = item["hint_level"]
            state["hint_count"] += 1
        independent = move in {"PROBE", "CHECK_MASTERY"}
        plan = _build_plan(
            scenario=scenario,
            episode_id=sid,
            turn=state["teaching_turns"],
            current=state["current_concept_id"],
            goal=state["goal_concept_id"],
            navigation_action_id=state["navigation_action_id"],
            teaching_action_id=action,
            active_item=item,
            beliefs=beliefs,
            hint_level=item["hint_level"] if item else 0,
            previous_step=f"{state['previous_move']} on {state['current_concept_id']}",
            assessment_independent=independent,
            actor_eligible=not forced,
            forced_answer=forced,
        )
        plan.update(
            plan_id=f"{sid}-plan-{count}",
            trace_id=trace_id,
            based_on_state_version=state["state_version"],
            expires_at=(
                self._now()
                + timedelta(seconds=self.config.get("pending_plan_timeout_seconds", 600))
            ).isoformat(),
        )
        plan["versions"]["strategist"] = self.strategist.policy_version
        state.update(
            stack=stack.stack,
            visit_index=stack.visit_index,
            last_pair_visit=[[list(k), v] for k, v in stack.last_pair_visit.items()],
            pair_visits=[[list(k), v] for k, v in stack.pair_visits.items()],
        )
        state["requests"]["answer"] = False
        return plan, state["active_item"]

    def commit_plan(self, body, idempotency_key, *, trace_id):
        def commit(db):
            state = self._session(db, body["session_id"])
            row = self._plan(db, body["plan_id"])
            if row["session_id"] != state["session_id"]:
                self._fail("INVALID_SCHEMA", "Plan does not belong to session", state)
            if row["status"] == "committed":
                if row["commit_body_hash"] == _hash(body):
                    return json.loads(row["commit_response_json"])
                self._fail("STATE_CONFLICT", "Plan already committed with a different body", state)
            self._expire(db, row)
            if row["canceled_reason"] == "expired":
                return ServiceError(
                    "PLAN_EXPIRED", "Plan has expired", current_state_version=state["state_version"]
                )
            if row["status"] != "pending":
                self._fail("STATE_CONFLICT", "Plan is not pending", state)
            plan = json.loads(row["plan_json"])
            if body["based_on_state_version"] != plan["based_on_state_version"] or (
                body["based_on_state_version"] != state["state_version"]
            ):
                self._fail("STATE_CONFLICT", "State version is stale", state)
            if body["effective_move"] != plan["teaching_action"]:
                self._fail("INVALID_SCHEMA", "Effective move must equal the reserved action", state)
            state = json.loads(row["state_json"])
            move = plan["teaching_action"]["move"]
            state.update(
                state_version=state["state_version"] + 1,
                teaching_turns=state["teaching_turns"] + 1,
                option_turns=state["option_turns"] + 1,
                last_response_id=body["response_id"],
                last_plan=plan,
                previous_move=move,
            )
            if move == "PROBE":
                state["no_diagnostic_yet"] = False
            if move == "CHECK_MASTERY" and state["skip_deadline"] is not None:
                state["skip_diagnostic_done"] = True
            result = dict(
                schema_version="1.0",
                plan_id=plan["plan_id"],
                status="committed",
                state_version=state["state_version"],
                public_state=self._public(state),
            )
            self._save(db, state)
            db.execute(
                "UPDATE plans SET status='committed', content_hash=?, response_id=?, "
                "commit_body_hash=?, commit_response_json=? WHERE plan_id=?",
                (
                    body["content_hash"],
                    body["response_id"],
                    _hash(body),
                    canonical(result),
                    plan["plan_id"],
                ),
            )
            self._event(db, state["session_id"], "teaching_committed", body)
            return result

        return self._post(
            "commit", body, idempotency_key, "CommitPlanRequest", "CommitPlanResponse", commit
        )

    @staticmethod
    def _status(row):
        return dict(
            schema_version="1.0",
            **{
                k: row[k]
                for k in ("plan_id", "session_id", "status", "content_hash", "response_id")
            },
        )

    def get_plan(self, plan_id, *, trace_id):
        with self.store.transaction() as db:
            row = self._plan(db, plan_id)
            self._expire(db, row)
            result = self._status(row)
            schemas.validate("PlanStatus", result)
            return result

    def cancel_plan(self, body, *, trace_id):
        def cancel(db):
            state = self._session(db, body["session_id"])
            row = self._plan(db, body["plan_id"])
            if row["session_id"] != state["session_id"]:
                self._fail("INVALID_SCHEMA", "Plan does not belong to session", state)
            if row["status"] == "committed":
                self._fail("STATE_CONFLICT", "Committed plans cannot be canceled", state)
            self._expire(db, row)
            if row["status"] == "pending":
                db.execute(
                    "UPDATE plans SET status='canceled', canceled_reason='requested' "
                    "WHERE plan_id=?",
                    (row["plan_id"],),
                )
                row["status"] = "canceled"
                self._event(db, state["session_id"], "plan_canceled", {"plan_id": row["plan_id"]})
            return self._status(row)

        return self._post("cancel", body, _request_id(body), "CancelRequest", "PlanStatus", cancel)

    def grade(self, body, *, trace_id):
        def grade(db):
            correctness = "unknown"
            for row in db.execute("SELECT item_json FROM plans WHERE item_json IS NOT NULL"):
                data = json.loads(row[0])
                if data and data["item_id"] == body["problem_instance_id"]:
                    item = self._item(data)
                    correctness = grade_answer(
                        body["normalized_answer"], item["instance"], item["family"]
                    )
                    break
            return dict(
                schema_version="1.0",
                problem_instance_id=body["problem_instance_id"],
                correctness=correctness,
                grade_source="deterministic" if correctness != "unknown" else "none",
            )

        return self._post("grade", body, _request_id(body), "GradeRequest", "GradeResponse", grade)
