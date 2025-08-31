apt-get update && apt-get install -y git

# This is only required for development
git config --global --add safe.directory /home/ubuntu/subwaysurfersai/workspace

git clone https://github.com/Genymobile/scrcpy
cd scrcpy

# TODO: Maybe manually build it later?
./install_release.sh

# For some reason this does not install via requirements.py
pip install pure-python-adb opencv-python-headless grpcio grpcio-tools

#sudo apt-get install xvfb -y