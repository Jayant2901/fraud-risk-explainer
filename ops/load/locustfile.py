# Load test for the synchronous scoring path.
#
#   pip install -r requirements-dev.txt   # includes locust
#   API_KEY=<your key> locust -f ops/load/locustfile.py --host http://localhost:8000
#
# Headless, for a fixed run (what docs/performance.md's numbers came
# from):
#   API_KEY=<your key> locust -f ops/load/locustfile.py --host http://localhost:8000 \
#       --headless --users 20 --spawn-rate 5 --run-time 60s --csv ops/load/results
#
# POST /api/score-custom, not POST /api/score: score-custom needs only
# TransactionAmt (see api/main.py's CustomTransactionRequest) so this
# runs against any deployment without needing the IEEE-CIS sample
# dataset seeded first — the real deployment target (real transactions
# arriving with real amounts), not the demo's ~30 cached entities.
import os
import random

from locust import HttpUser, between, task

API_KEY = os.environ.get("API_KEY", "")
PRODUCT_CODES = ["W", "C", "R", "H", "S"]
CARD_NETWORKS = ["visa", "mastercard", "american express", "discover"]


class RiskManagerUser(HttpUser):
    # A person or an upstream payment processor firing transactions, not
    # a tight polling loop — small jitter between requests per user.
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(10)
    def score_custom(self):
        payload = {
            "TransactionAmt": round(random.uniform(5, 5000), 2),
            "ProductCD": random.choice(PRODUCT_CODES),
            "card4": random.choice(CARD_NETWORKS),
            "hour_of_day": random.randint(0, 23),
        }
        self.client.post("/api/score-custom", json=payload, headers=self.headers, name="/api/score-custom")

    @task(1)
    def health(self):
        # Unauthenticated on purpose (see api/main.py) — exercised here
        # as the cheap liveness check a load balancer would actually hit
        # repeatedly, not to stress the scoring path.
        self.client.get("/api/health", name="/api/health")
