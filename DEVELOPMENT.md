# ReFinED Development
We recommend developing code on an EC2 instance with a GPU. We use p3.8xlarge as it has a large amount of memory and 4 GPUs which enables fast training.

- Copy the ReFinED code to your machine `git clone https://github.com/amazon-science/ReFinED.git`.
- Use your preferred IDE (such as PyCharm) to modify the code.
- Copy the ReFinED code to your EC2 instance `rsync -zarv --prune-empty-dirs --include "*/" --include "*.py" --exclude="*" ReFinED/ ubuntu@X.XXX.XXX.XX:/path_to_code/`.
- Add a new host to your ssh config file so that you do not need to provide the path to the private ssh key each time you connect or copy data (~/.ssh/config) example:
```
Host x.xxx.xxx.xxx
     User ubuntu
     IdentityFile <path_to_ssh_private_key_file>
```
- SSH into your EC2 instance.
- Install the Python dependencies using `pip install -r requirements.txt`.
- Add ReFinED to your Python path `export PYTHONPATH=$PYTHONPATH:/<path_to_code>/src/`.
- To confirm the setup is correct run `python example_scripts/refined_demo.py` runs successfully.

