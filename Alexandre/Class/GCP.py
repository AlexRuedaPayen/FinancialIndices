import paramiko,scp


class GCP:

    def __init__(self,
                host="34.68.238.69",
                user="MacAlexandre_GCP_VM1",
                keyfile="/Users/alexandreprofessional2/Desktop/key/key_GCP_VM1",
                class_=["Stock"],
                data_=["RUI.PA"]):
        ssh = paramiko.SSHClient()
        self.ssh=ssh 
        self.host=host
        self.user=user
        from getpass import getpass
        password = getpass()
        k = paramiko.RSAKey.from_private_key_file(keyfile,password=password)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=user, pkey=k)
        self.class_=class_
        """
        stdin,stdout,stderr=ssh.exec_command('pwd')
        print(stdout.read().decode('utf8'))
        #print(stdin.read().decode('utf-8'),stdout.read().decode('utf-8'))
        stdin.close()
        stdout.close()
        stderr.close()"""

    def __enter__(self):
        file_path_origin="./Alexandre/Class/"
        file_path_destination="~/Projects/Financial_Indices/Alexandre/Class/"
        self.scp=scp.SCPClient(self.ssh.get_transport())
        for i in self.class_:
            file_path_class_origin=file_path_origin+i+".py"
            file_path_class_destination=self.user+"@instance-2:"+file_path_destination+i+".py"
            self.scp.put(file_path_class_origin,file_path_class_destination)
            print("File "+file_path_class_destination+" created")
        return(self)

    def run(self,script):
        file_path_script_origin="./Alexandre/script/"+script
        file_path_script_destination="~/Projects/Financial_Indices/Alexandre/script/"+script
        subprocess.run(["scp", file_path_script_origin, "USER@SERVER:"+file_path_script_destination])
        subprocess.run(["python3", file_path_script_destination])


    def __exit__(self):
        for i in file:
            subprocess.run(["scp", "USER@SERVER:"+i,i])
        self.ssh.disconnect()

a=GCP()
a.__enter__()