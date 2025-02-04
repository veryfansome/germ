from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from settings import germ_settings


def setup_tracing(service_name: str):
    resource = Resource.create({
        "service.name": "model-service",
    })
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    otlp_exporter = OTLPSpanExporter(
        endpoint=f"{germ_settings.JAEGER_HOST}:{germ_settings.JAEGER_PORT}",
        insecure=True,
    )

    span_processor = BatchSpanProcessor(otlp_exporter)

    provider.add_span_processor(span_processor)
