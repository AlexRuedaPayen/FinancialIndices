import paramiko,scp

class GCP:

    def __init__(self,
                host="34.68.238.69",
                user="MacAlexandre_GCP_VM1",
                keyfile="/Users/alexandreprofessional2/Desktop/key/key_GCP_VM1",
                class_=["Stock"],
                data_=["RUI.PA"]):

        self.host=host
        self.user=user
        self.class_=class_

        from getpass import getpass
        password = getpass()

        ssh = paramiko.SSHClient()
        k = paramiko.RSAKey.from_private_key_file(keyfile,password=password)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=user, pkey=k)

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
        print("Function "+file_path_script_destination+" done")
        self.ssh.exec_command('conda deactivate')
        """self.scp.get(file_path_script_origin,file_path_script_destination)"""


    def __exit__(self,type, value, traceback):
        self.ssh.close()
