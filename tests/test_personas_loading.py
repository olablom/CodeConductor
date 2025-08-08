from codeconductor.debate.personas import load_personas_yaml, build_agents_from_personas


def test_load_and_build_personas_default():
    personas = load_personas_yaml(None)
    assert "architect" in personas and "coder" in personas
    agents = build_agents_from_personas(personas, roles=["architect", "coder"])
    names = [a.name for a in agents]
    assert names == ["Architect", "Coder"]
    assert any("Guidelines" in a.persona for a in agents)
