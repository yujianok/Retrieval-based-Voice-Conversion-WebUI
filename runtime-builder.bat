mkdir runtime
cd runtime
curl -O https://bootstrap.pypa.io/get-pip.py
.\python.exe get-pip.py
.\python.exe -m pip install --upgrade pip==24.0
.\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\python.exe -m pip install -r ../requirements.txt