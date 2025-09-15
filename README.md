> :warning: Warning! This project is in its very early stages of development. Expect frequent changes and potential breaking updates as we iterate on features and architecture.

Installation
The recommended way to install the AI Agent is by using the provided Helm chart.

1. Clone the repository:

2. Configure Container Registry Access:

The container images are hosted in a private GitHub Container Registry (ghcr.io). You need to provide your GitHub credentials to allow Kubernetes to pull the images.

Create a Kubernetes secret named gh-secret with your credentials:

```
kubectl create secret docker-registry gh-secret \
  --docker-server=ghcr.io \
  --docker-username=<YOUR_GITHUB_USERNAME> \
  --docker-password=<YOUR_PERSONAL_ACCESS_TOKEN> \
  --docker-email=<YOUR_EMAIL> \
  --namespace cattle-ai-agent-system
```

Note: For the docker-password, you must use a Personal Access Token (PAT) from your GitHub account, not your regular password. The PAT needs the read:packages scope.

3. Install with Helm:

Finally, use Helm to install the agent from the local chart directory. This command will create the cattle-ai-agent-system namespace if it doesn't already exist. 

```bash
helm install ai-agent chart --namespace cattle-ai-agent-system --create-namespace --set "imagePullSecrets[0].name=gh-secret" --set ollamaUrl=http://ollama-url --set model=qwen3:1.7b
```

The previous installation command configures Ollama. To use OpenAI or Gemini, you can either:

- Edit the llm-settings secret after installation. This require to restart the pod.
- Supply different values when installing the chart.

The temporary UI can be accessed in https://your-rancher-url/api/v1/namespaces/cattle-ai-agent-system/services/http:rancher-ai-agent:80/proxy/agent
This UI will be replaced by the UI extension




