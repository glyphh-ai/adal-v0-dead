"""The HTTP API, as the Glyphh server calls it."""

FACT = "Piheba Fupu's account manager is Nipira."
ASK = {"messages": [{"role": "user", "content": "Who is the account manager for Piheba Fupu?"}]}


def test_health_needs_no_token(client):
    r = client(token="").get("/v1/health")
    assert r.status_code == 200 and r.json()["backend"] == "fake"


def test_a_wrong_token_or_a_bad_tenant_is_refused(client):
    assert client(token="nope").get("/v1/facts").json()["error"]["code"] == "E_UNAUTHORIZED"
    assert client(tenant="No Such Tenant!").get("/v1/facts").json()["error"]["code"] == "E_INVALID"
    assert client(tenant="").get("/v1/facts").status_code == 422


def test_tell_list_ask_forget(client):
    c = client()
    told = c.post("/v1/facts", json={"text": FACT, "source": "chat"})
    assert told.status_code == 201
    fact = told.json()["facts"][0]
    assert fact["text"] == FACT and fact["source"] == "chat"
    assert c.get("/v1/facts").json() == {"facts": [fact], "next": None}
    reply = c.post("/v1/chat", json=ASK).json()
    assert "Nipira" in reply["reply"] and reply["memory"]["fired"] > 0
    assert c.delete(f"/v1/facts/{fact['id']}").status_code == 204
    assert c.delete(f"/v1/facts/{fact['id']}").json()["error"]["code"] == "E_NOT_FOUND"
    assert c.post("/v1/chat", json=ASK).json()["reply"] == "I do not know."


def test_search_and_pages(client):
    c = client()
    c.post("/v1/facts", json={"texts": [f"Person{chr(65 + i)} Doti lives in Huraku." for i in range(5)]})
    c.post("/v1/facts", json={"text": FACT})
    first = c.get("/v1/facts", params={"limit": 4}).json()
    assert len(first["facts"]) == 4 and first["next"]
    rest = c.get("/v1/facts", params={"limit": 4, "after": first["next"]}).json()
    assert len(rest["facts"]) == 2 and rest["next"] is None
    assert [f["text"] for f in c.get("/v1/facts", params={"q": "nipira"}).json()["facts"]] == [FACT]


def test_one_tenant_cannot_read_or_forget_anothers(client):
    fact = client("acme").post("/v1/facts", json={"text": FACT}).json()["facts"][0]
    other = client("globex")
    assert other.get("/v1/facts").json()["facts"] == []
    assert other.delete(f"/v1/facts/{fact['id']}").status_code == 404
    assert other.post("/v1/chat", json=ASK).json()["reply"] == "I do not know."
    assert other.delete("/v1/facts").status_code == 204  # theirs, which is empty
    assert len(client("acme").get("/v1/facts").json()["facts"]) == 1


def test_malformed_requests(client):
    c = client()
    assert c.post("/v1/facts", json={}).json()["error"]["code"] == "E_INVALID"
    assert c.post("/v1/chat", json={"messages": []}).json()["error"]["code"] == "E_INVALID"
    assert c.post("/v1/chat", json={"nope": 1}).json()["error"] == {
        "code": "E_INVALID", "message": "The request is malformed.", "field": "messages",
    }  # fmt: skip
    assert c.post("/v1/ingest", json={"text": " "}).json()["error"]["code"] == "E_INVALID"
