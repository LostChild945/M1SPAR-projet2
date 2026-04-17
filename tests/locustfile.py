"""
Locust load testing — API de recommandation P2

Scénario : 100 users simultanés pendant 60s.
Objectif : P95 < 50ms sur GET /recommend.

Usage :
    locust -f tests/locustfile.py --headless -u 100 -r 10 \
           --run-time 60s --html reports/locust_p2_100users.html
"""

from locust import HttpUser, task, between


class RecoUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = "http://localhost:8000"

    @task(3)
    def recommend(self):
        self.client.get(
            "/recommend/USER_001",
            params={"n": 10, "exclude_purchased": True},
            name="/recommend/{user_id}",
        )

    @task(2)
    def recommend_with_category(self):
        self.client.get(
            "/recommend/USER_002",
            params={"n": 5, "category": "All_Beauty"},
            name="/recommend/{user_id}?category",
        )

    @task(2)
    def similar(self):
        self.client.get(
            "/similar/PROD_001",
            params={"n": 5},
            name="/similar/{product_id}",
        )

    @task(1)
    def feedback(self):
        self.client.post(
            "/feedback",
            json={
                "user_id": "USER_001",
                "product_id": "PROD_001",
                "interaction_type": "click",
                "ab_variant": "control",
            },
        )

    @task(1)
    def health(self):
        self.client.get("/health")
