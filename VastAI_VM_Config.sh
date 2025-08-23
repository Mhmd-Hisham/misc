###################### CONFIGURE VAST.AI VM BASE DOCKER IMAGE ######################
# to suppress interactive "services to restart" prompt
export NEEDRESTART_MODE=a
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
# nvidia container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
sudo apt-get install zip python3-pip python-is-python3 -y 
pip install -U "huggingface_hub[cli]"

wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb
sudo dpkg -i nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb

sudo apt-get -y remove --purge 'nvidia-*'
sudo apt-get autoremove -y
sudo apt-get clean
sudo apt-get update
sudo apt-get -y install cuda-drivers

sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# configure docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# enable ncu
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-perf.conf
sudo update-initramfs -u
sudo reboot

