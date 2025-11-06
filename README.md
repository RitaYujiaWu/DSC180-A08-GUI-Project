# DSC180-A08-GUI-Project

## Part II — Online Data Evaluation (Based on ZeroGUI's framework)
📝 **Status:** AndroidWorld environment is interactive; adapter wired. OSWorld/AndroidLab are inherently compatible. \
🎯 **Goal:** Integrate AndroidWorld into online evaluation by running it inside Docker while exposing a ZeroGUI-compatible environment. \
📈 **Next Step:** Try evaluation with the env that we've set up.

### TL;DR
-	We run AndroidWorld in a Docker container.
-	We added an adapter android_world_env.py so ZeroGUI can call the env in its usual way.
-	You can smoke-test the interaction in notebooks/docker_exp.ipynb.

### 1. Project Layout
`git clone` ZeroGUI's [repo](https://github.com/OpenGVLab/ZeroGUI) under the root and move the files under `/online_eval` folder to the designated place as shown:
<pre><code>repo-root/
  zerogui/
    openrlhf/
      env/
        __init__.py
        ...
        osworld_env.py
        android_lab_env.py
        android_world_env.py     # <-- our adapter (new)
    ...
    docker_exp.ipynb             # <-- end-to-end sanity test
</code></pre>
### 2. Build / Run the AndroidWorld Container
Build and run the Docker container following AndroidWorld's [🔗Docker setup](https://github.com/google-research/android_world?tab=readme-ov-file#docker-support-experimental) \
Once inside the container:
<pre><code>cd /workspace/zerogui
pip install -r requirements.txt (Please use Python >=3.10)</code></pre>
### 3. Smoke Test
Open and run the notebook `docker_exp.ipynb` \
You could play with the action by changing `action_payload`
<pre><code>action_payload = {"action_type": "open_app", "app_name": "Chrome"}
</code></pre>
And you should see the corresponding screenshots/empty reward of your interactions.
