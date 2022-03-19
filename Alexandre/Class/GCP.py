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
            file_path_class_destination=file_path_destination#+i+".py" #self.user+"@instance-2:"+
            self.scp.put(file_path_class_origin,file_path_class_destination)
            print("File "+file_path_class_destination+" created")
        return(self)

    def run(self,script):
        file_path_script_origin="./Alexandre/script/"+script+".py"
        file_path_script_destination="~/Projects/Financial_Indices/Alexandre/script/"+script+".py"
        self.scp.put(file_path_script_origin,file_path_script_destination)
        """self.ssh.exec_command('conda source activate Financial_Indices')
        self.ssh.exec_command('python3 '+file_path_script_destination)
        self.scp.get(file_path_script_origin,file_path_script_destination)"""


    def __exit__(self,type, value, traceback):
        pass
