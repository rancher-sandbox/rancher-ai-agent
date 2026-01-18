import logging
import os
import certifi

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from .services.memory import create_memory_manager
from .routers import chat, websocket, ui

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# This will be removed once https://github.com/modelcontextprotocol/python-sdk/pull/1177 is merged
class SimpleTruststore:
    def get_default(self):
        """Get the default Python truststore"""
        return certifi.where()

    def create_combined(self, company_cert_path, output_path):
        """Create truststore with public CAs + company cert"""
        with open(output_path, "w") as combined:
            # Add public CAs 
            with open(certifi.where(), "r") as public_cas:
                combined.write(public_cas.read())

            # Add MCP self-signed cert
            with open(company_cert_path, "r") as company:
                combined.write("\n" + company.read())

        return output_path
    
    def use_truststore(self, truststore_path):
        """Set the global truststore"""
        os.environ["SSL_CERT_FILE"] = truststore_path

    def set_truststore(self):
        company_cert_path = "/etc/tls/tls.crt"
        output_path = "/combined.crt"

        if os.path.exists(company_cert_path):
            truststore_path = self.create_combined(
                company_cert_path=company_cert_path, output_path=output_path
            )
            self.use_truststore(truststore_path=truststore_path)
        else:
            logging.warning(f"Company cert not found at {company_cert_path}, skipping truststore setup.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
        logging.getLogger().setLevel(LOG_LEVEL)
        if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != "true":
            SimpleTruststore().set_truststore()

        app.memory_manager = await create_memory_manager()
    except ValueError as e:
        logging.critical(e)
        raise e
    yield
    await app.memory_manager.destroy()

app = FastAPI(lifespan=lifespan)

app.include_router(websocket.router)
app.include_router(chat.router)

if os.environ.get("ENABLE_TEST_UI", "").lower() == "true":
    app.include_router(ui.router)