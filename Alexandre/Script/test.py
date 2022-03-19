from Alexandre.Class.GCP import GCP

if __name__ == '__main__':
    with GCP() as f:
        f.run("fun_A")