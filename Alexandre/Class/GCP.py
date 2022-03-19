class GCP:

    import paramiko

    def __init__(self,host="MacAlexandre@34.125.182.253",
                 keyfile="~/.ssh/VM-1-GCP-Instance1/key",
                 class_=["Stock"],
                 data_=["RUI.PA"]):
        ssh = paramiko.SSHClient()
        self.ssh=ssh 
        k = paramiko.RSAKey.from_private_key_file(keyfile)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username="AlexRuedaPayen", pkey=k)
        self.class_=class_

    def __enter__(self):
        file_path_origin="./Alexandre/class/"
        file_path_destination="~/Projects/Financial_Indices/Alexandre/class/"
        for i in self.class_:
            file_path_class_origin=file_path_origin+i
            file_path_class_destination=file_path_destination+i
            subprocess.run(["scp", file_path_class_origin, "USER@SERVER:"+file_path_class_destination])
        return()

    def run(self,script):
        file_path_script_origin="./Alexandre/script/"+script
        file_path_script_destination="~/Projects/Financial_Indices/Alexandre/script/"+script
        subprocess.run(["scp", file_path_script_origin, "USER@SERVER:"+file_path_script_destination])
        subprocess.run(["python3", file_path_script_destination])


    def __exit__(self):
        for i in file:
            subprocess.run(["scp", "USER@SERVER:"+i,i])
        self.ssh.disconnect()
