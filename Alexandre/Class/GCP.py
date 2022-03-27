import paramiko,scp,json

class GCP:

    def __init__(self,
                keyfile="/Users/alexandreprofessional2/Desktop/key/key_GCP_VM1",
                class_=["Asset"],
                data_=["RUI.PA"]):

        with open('/Users/alexandreprofessional2/Desktop/key/GCP/credentials.json', 'r') as f:
            credentials = json.load(f)

        self.host=credentials['host']
        self.user=credentials['user']
        self.class_=class_

        from getpass import getpass
        password = getpass()

        ssh = paramiko.SSHClient()
        k = paramiko.RSAKey.from_private_key_file(keyfile,password=password)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=self.host, username=self.user, pkey=k)

        self.ssh=ssh 
        self.scp=scp.SCPClient(self.ssh.get_transport())

    def __enter__(self):
        file_path_origin="./Alexandre/Class/"
        file_path_destination="~/Projects/Financial_Indices/Alexandre/Class/"
        for i in self.class_:
            file_path_class_origin=file_path_origin+i+".py"
            file_path_class_destination=file_path_destination
            self.scp.put(file_path_class_origin,file_path_class_destination)
            print("File "+file_path_class_destination+i+".py"+" created")
        return(self)

    def run(self,script):
        file_path_script_origin="./Alexandre/Script/"+script+".py"
        file_path_script_destination="~/Projects/Financial_Indices/Alexandre/Script/"+script+".py"
        self.scp.put(file_path_script_origin,file_path_script_destination)
        print("File "+file_path_script_destination+" created")
        self.ssh.exec_command('conda activate')
        print("Running function "+file_path_script_destination)
        stdin,stdout,stderr=self.ssh.exec_command('python3.7 '+file_path_script_destination)
        stdin.close()
        stdout.close()
        stderr.close()
        print("Function "+file_path_script_destination+" done")
        self.ssh.exec_command('conda deactivate')
        self.ssh.exec_command('rm file_path_script_destination')
        file_path_result_origin="~/Projects/Financial_Indices/Data/Script/fun_A.json"
        file_path_result_destination="./Alexandre/Script/fun_A.json"
        self.scp.get(file_path_result_origin,file_path_result_destination)


    def __exit__(self,type, value, traceback):
        self.ssh.close()

with GCP() as f:
    f.run("fun_A")
