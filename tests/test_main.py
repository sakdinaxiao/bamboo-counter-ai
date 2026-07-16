def test_main_import():
    try:
        import src.main
    except Exception as e:
        assert False, f"Importing main failed with error: {e}"
