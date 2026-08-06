class Postprocessor:
    """Converts pipeline results into final text output."""

    def format_output(self, result: dict) -> str:
        return result["prediction"]
