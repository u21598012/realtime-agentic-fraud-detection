from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Any
import xgboost as xgb
import json

class AgentState(TypedDict, total=False):
    input: str
    transaction: dict
    actual: int
    predicted: int
    decision: int
    retry: int
    outcome_reason: str
    route: Literal["reject", "human_verification", "fraud_detection", "accept"]
    trace: list[str]


class Transaction_Inspector:
    def __init__(self):
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model("fraud_detection_xgb_model.json")
        self.llm = Ollama(model="llama3.1")
        self.feature_cols = [
            "step",
            "amount",
            "oldbalanceorg",
            "newbalanceorig",
            "oldbalancedest",
            "newbalancedest",
            "balancechangeorig",
            "balancechangedest",
            "errorbalanceorig",
            "errorbalancedest",
            "issameuser",
            "type_encoded",
        ]
        self.graph = self._build_graph()

    def _add_trace(self, state: AgentState, node: str) -> list[str]:
        trace = state.get("trace", []) or []
        return [*trace, node]
    
    def _splitter(self, state: AgentState) -> AgentState:
        tx = state.get("transaction")
        if tx is None:
            try:
                tx = json.loads(state.get("input", ""))
            except Exception:
                return {
                    "input": state.get("input", ""),
                    "route": "reject",
                    "transaction": {},
                    "retry": state.get("retry", 0),
                    "actual": state.get("actual"),
                    "outcome_reason": "invalid transaction format",
                    "trace": self._add_trace(state, "splitter"),
                }

        amount = float(tx.get("amount", 0) or 0)
        if amount <= 0:
            return {
                "input": state.get("input", ""),
                "route": "reject",
                "transaction": tx,
                "retry": state.get("retry", 0),
                "actual": state.get("actual"),
                "outcome_reason": "invalid amount",
                "trace": self._add_trace(state, "splitter"),
            }

        llm_input = (
            f"Transaction details: {tx}. Should this transaction be sent for human verification "
            f"or fraud detection? Respond with 'human_verification' or 'fraud_detection'."
        )
        try:
            llm_response = self.llm.invoke(llm_input)
            route_decision = str(llm_response).strip().lower()
        except Exception:
            route_decision = "fraud_detection"

        if route_decision not in ["human_verification", "fraud_detection"]:
            route_decision = "fraud_detection"

        return {
            "input": state.get("input", ""),
            "route": route_decision,
            "transaction": tx,
            "retry": state.get("retry", 0),
            "actual": state.get("actual"),
            "outcome_reason": "",
            "trace": self._add_trace(state, "splitter"),
        }

    def human_verification(self, state: AgentState) -> AgentState:
        actual = state.get("actual")
        if actual is None:
            return {
                **state,
                "route": "reject",
                "outcome_reason": state.get("outcome_reason", "") + " | missing ground truth",
                "trace": self._add_trace(state, "human_verification"),
            }

        if actual == 1:
            return {
                "input": state.get("input", ""),
                "route": "reject",
                "transaction": state.get("transaction", {}),
                "retry": state.get("retry", 0),
                "actual": actual,
                "outcome_reason": "human verification failed",
                "trace": self._add_trace(state, "human_verification"),
            }

        return {
            "input": state.get("input", ""),
            "route": "accept",
            "transaction": state.get("transaction", {}),
            "retry": state.get("retry", 0),
            "actual": actual,
            "outcome_reason": "human verification successful",
            "trace": self._add_trace(state, "human_verification"),
        }
            
    def fraud_detection(self, state: AgentState) -> AgentState:
        try:
            features = self._extract_features(state.get("transaction", {}))
            classifier_result = self.classifier.predict(features)
            predicted = int(classifier_result[0])
        except Exception:
            predicted = 1

        if predicted == 1:
            return {
                "input": state.get("input", ""),
                "route": "reject",
                "transaction": state.get("transaction", {}),
                "retry": state.get("retry", 0),
                "actual": state.get("actual"),
                "predicted": predicted,
                "outcome_reason": "system identified fraud",
                "trace": self._add_trace(state, "fraud_detection"),
            }

        return {
            "input": state.get("input", ""),
            "route": "human_verification",
            "transaction": state.get("transaction", {}),
            "retry": state.get("retry", 0),
            "actual": state.get("actual"),
            "predicted": predicted,
            "outcome_reason": "system did not identify, routing to human verification",
            "trace": self._add_trace(state, "fraud_detection"),
        }

    def reject(self, state: AgentState) -> AgentState:
        return {
            **state,
            "decision": 0,
            "actual": state.get("actual"),
            "predicted": state.get("predicted"),
            "outcome_reason": state.get("outcome_reason", "") + " | final decision: reject",
            "trace": self._add_trace(state, "reject"),
        }
    
    def accept(self, state: AgentState) -> AgentState:
        return {
            **state,
            "decision": 1,
            "actual": state.get("actual"),
            "predicted": state.get("predicted"),
            "outcome_reason": state.get("outcome_reason", "") + " | final decision: accept",
            "trace": self._add_trace(state, "accept"),
        }

    def _route_from_splitter(self, state: AgentState) -> str:
        return state.get("route", "reject")

    def _extract_features(self, tx: dict) -> Any:
        row = [[tx.get(col, 0) for col in self.feature_cols]]
        return row

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("splitter", self._splitter)
        graph.add_node("human_verification", self.human_verification)
        graph.add_node("fraud_detection", self.fraud_detection)
        graph.add_node("reject", self.reject)
        graph.add_node("accept", self.accept)

        graph.set_entry_point("splitter")
        graph.add_conditional_edges(
            "splitter",
            self._route_from_splitter,
            {
                "reject": "reject",
                "human_verification": "human_verification",
                "fraud_detection": "fraud_detection",
            },
        )
        graph.add_conditional_edges(
            "human_verification",
            lambda state: state.get("route", "reject"),
            {
                "reject": "reject",
                "accept": "accept",
            },
        )
        graph.add_conditional_edges(
            "fraud_detection",
            lambda state: state.get("route", "reject"),
            {
                "reject": "reject",
                "human_verification": "human_verification",
            },
        )
        graph.add_edge("reject", END)
        graph.add_edge("accept", END)

        return graph.compile()
    
    def execute(self, transaction: Any, actual: int | None = None) -> AgentState:
        inferred_actual = None
        if actual is None and isinstance(transaction, dict) and "isfraud" in transaction:
            try:
                inferred_actual = int(transaction.get("isfraud"))
            except Exception:
                inferred_actual = None

        input_state: AgentState = {
            "input": json.dumps(transaction) if isinstance(transaction, dict) else str(transaction),
            "transaction": transaction if isinstance(transaction, dict) else None,
            "actual": actual if actual is not None else inferred_actual,
            "retry": 0,
            "trace": [],
        }
        final_state = self.graph.invoke(input_state)
        return final_state


if __name__ == '__main__':
    engine = Transaction_Inspector()

    sample_tx = {
        "step": 355,
        "amount": 129493.82,
        "oldbalanceorg": 496957.0,
        "newbalanceorig": 367463.18,
        "oldbalancedest": 0.0,
        "newbalancedest": 129493.82,
        "balancechangeorig": 129493.82,
        "balancechangedest": 129493.82,
        "errorbalanceorig": 0.0,
        "errorbalancedest": 0.0,
        "issameuser": False,
        "type_encoded": 1,
    }

    result = engine.execute(sample_tx, actual=0)
    print("Decision state:", result)