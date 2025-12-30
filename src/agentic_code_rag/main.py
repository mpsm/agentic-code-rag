from strands.telemetry import StrandsTelemetry

from .cli import cli

StrandsTelemetry().setup_otlp_exporter().setup_meter(enable_otlp_exporter=True)

if __name__ == "__main__":
    cli()
