from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

import requests
import time

MCP_IMAGE_NAME = "ghcr.io/rancher-sandbox/rancher-ai-mcp:v0.0.1-alpha.25" 
EXPOSED_PORT = 9092 

def test_agent():
    with DockerContainer(MCP_IMAGE_NAME).with_env("INSECURE_SKIP_TLS", "true").with_exposed_ports(EXPOSED_PORT) as container:
        container.start()
        wait_for_logs(container, "MCP Server started!")

        host_port = container.get_exposed_port(EXPOSED_PORT)
        host_ip = container.get_container_host_ip()
        service_url = f"http://{host_ip}:{host_port}"
        
        print(f"Custom Service is running at: {service_url}")
        response = requests.get(service_url)
        print(response.text)
        
        # Now you can use a request library (like `requests`) to interact 
        # with your service and run your tests.
        # Example:
        # response = requests.get(f"{service_url}/api/data")
        # assert response.status_code == 200
        
    # Container is automatically stopped and removed here