def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_customer_ok(client):
    response = client.get("/customers/NEO-88213", params={"phone": "0612840193"})
    assert response.status_code == 200
    body = response.json()
    assert body["full_name"] == "Camille Rousseau"
    assert body["postal_code"] == "75019"


def test_get_customer_wrong_phone_is_forbidden(client):
    response = client.get("/customers/NEO-88213", params={"phone": "0000000000"})
    assert response.status_code == 403


def test_get_customer_not_found(client):
    response = client.get("/customers/NEO-00000", params={"phone": "0612840193"})
    assert response.status_code == 404


def test_get_customer_invoices(client):
    response = client.get("/customers/NEO-88213/invoices", params={"phone": "0612840193"})
    assert response.status_code == 200
    invoices = response.json()
    assert len(invoices) == 3
    assert all(inv["status"] == "paid" for inv in invoices)


def test_list_incidents_filtered_by_postal_code(client):
    response = client.get("/incidents", params={"postal_code": "75019"})
    assert response.status_code == 200
    incidents = response.json()
    assert len(incidents) == 1
    assert incidents[0]["incident_id"] == "INC-4471"


def test_get_incident_not_found(client):
    response = client.get("/incidents/INC-9999")
    assert response.status_code == 404


def test_list_slots_available_only_by_default(client):
    response = client.get("/appointments/slots", params={"postal_code": "75019"})
    assert response.status_code == 200
    slots = response.json()
    slot_ids = {s["slot_id"] for s in slots}
    assert "SLOT-7A31" in slot_ids
    assert "SLOT-7A32" not in slot_ids  # available: false dans le fixture


def test_book_appointment_success_marks_slot_unavailable(client):
    response = client.post(
        "/appointments",
        json={"customer_id": "NEO-88213", "slot_id": "SLOT-7A31", "reason": "no_internet"},
    )
    assert response.status_code == 201
    appointment = response.json()
    assert appointment["status"] == "confirmed"
    assert appointment["customer_id"] == "NEO-88213"

    slots = client.get("/appointments/slots", params={"postal_code": "75019"}).json()
    assert all(s["slot_id"] != "SLOT-7A31" for s in slots)


def test_book_appointment_unknown_customer(client):
    response = client.post(
        "/appointments",
        json={"customer_id": "NEO-00000", "slot_id": "SLOT-7A31", "reason": "no_internet"},
    )
    assert response.status_code == 404


def test_book_appointment_unknown_slot(client):
    response = client.post(
        "/appointments",
        json={"customer_id": "NEO-88213", "slot_id": "SLOT-0000", "reason": "no_internet"},
    )
    assert response.status_code == 404


def test_book_appointment_already_unavailable_slot(client):
    # SLOT-7A32 est déjà `available: false` dans le fixture
    response = client.post(
        "/appointments",
        json={"customer_id": "NEO-88213", "slot_id": "SLOT-7A32", "reason": "no_internet"},
    )
    assert response.status_code == 409


def test_book_appointment_postal_code_mismatch(client):
    # NEO-88213 est à Paris (75019), SLOT-6B10 dessert Lyon (69007)
    response = client.post(
        "/appointments",
        json={"customer_id": "NEO-88213", "slot_id": "SLOT-6B10", "reason": "no_internet"},
    )
    assert response.status_code == 422


def test_cancel_appointment_frees_the_slot(client):
    booked = client.post(
        "/appointments",
        json={"customer_id": "NEO-88213", "slot_id": "SLOT-7A31", "reason": "no_internet"},
    ).json()

    cancel_response = client.delete(f"/appointments/{booked['appointment_id']}")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"

    slots = client.get("/appointments/slots", params={"postal_code": "75019"}).json()
    assert any(s["slot_id"] == "SLOT-7A31" for s in slots)


def test_cancel_appointment_twice_conflicts(client):
    booked = client.post(
        "/appointments",
        json={"customer_id": "NEO-88213", "slot_id": "SLOT-7A31", "reason": "no_internet"},
    ).json()

    client.delete(f"/appointments/{booked['appointment_id']}")
    second_cancel = client.delete(f"/appointments/{booked['appointment_id']}")
    assert second_cancel.status_code == 409


def test_create_ticket_and_fetch_it(client):
    response = client.post(
        "/tickets",
        json={
            "category": "technical",
            "summary": "Panne persistante malgré intervention technicien",
            "customer_id": "NEO-88213",
            "urgency": "high",
        },
    )
    assert response.status_code == 201
    ticket = response.json()

    fetched = client.get(f"/tickets/{ticket['ticket_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["summary"] == "Panne persistante malgré intervention technicien"


def test_create_ticket_unknown_customer(client):
    response = client.post(
        "/tickets",
        json={"category": "other", "summary": "test", "customer_id": "NEO-00000"},
    )
    assert response.status_code == 404
