import subprocess
import signal
import sys
import time
import requests
import webbrowser
import os

# Global variables to hold process identifiers or container IDs
orthanc_container_id = None
ohif_process = None
chromium_process = None

def is_service_ready(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def launch_orthanc():
    global orthanc_container_id
    process = subprocess.Popen(['docker', 'run', '-d', '-p', '8042:8042', '-p', '4242:4242', 'orthanc-image'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    orthanc_container_id = stdout.strip()
    print("Orthanc launched in container " + str(orthanc_container_id))

def stop_orthanc():
    global orthanc_container_id
    if orthanc_container_id:
        subprocess.run(['docker', 'stop', orthanc_container_id])
        print(f"Stopped Orthanc container {orthanc_container_id}.")

def launch_ohif():
    global ohif_process

    command = '/opt/start_monailabel.sh'

    ohif_process = subprocess.Popen(['bash', '-c', command])
    print("OHIF server started.")

def stop_ohif():
    global ohif_process
    if ohif_process:
        ohif_process.terminate()
        ohif_process.wait()
        print("OHIF server stopped.")

def launch_chromium():
    global chromium_process

    while not is_service_ready("http://127.0.0.1:8000/ohif/"):
        print("ohif not ready, waiting...")
        time.sleep(1)

    # webbrowser.open("http://127.0.0.1:8000/ohif")
    #chromium_process = subprocess.Popen(['chromium', "http://127.0.0.1:8000/ohif/"])
    chromium_process = subprocess.Popen(['chromium', "http://127.0.0.1:8000/ohif/viewer?StudyInstanceUIDs=1.76.380.18.10.1160420005004911.73"])
    print("Chromium launched and navigated to OHIF viewer.")


def stop_chromium():
    global chromium_process
    if chromium_process:
        chromium_process.terminate()
        chromium_process.wait()
        print("Chromium stopped.")

def signal_handler(sig, frame):
    print('Signal received, cleaning up...')
    stop_chromium()
    stop_ohif()
    stop_orthanc()
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        launch_orthanc()
        launch_ohif()
        #launch_chromium()
        # Keep the script running to listen for signals
        signal.pause()
    finally:
        # Ensure cleanup runs even if the script exits normally or an error occurs
        #stop_chromium()
        stop_ohif()
        #stop_orthanc()

if __name__ == '__main__':
    main()
