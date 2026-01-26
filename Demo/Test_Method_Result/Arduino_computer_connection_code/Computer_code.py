import serial
import time
from datetime import datetime
import psutil


arduino_port = "COM6"
baud_rate = 115200


try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to Arduino on {arduino_port} at {baud_rate} baud.")
    time.sleep(2)
except Exception as e:
    print(f"Error connecting to Arduino: {e}")




    exit()



def monitor_data_transfer(app_name, interval_ms):
    interval = interval_ms / 1000.0
    for proc in psutil.process_iter(['pid', 'name']):
        if app_name.lower() in proc.info['name'].lower():
            print(f"Monitoring Process: {proc.info['name']} (PID: {proc.info['pid']})")
            try:
                initial_counters = proc.io_counters()
                print(f"Initial I/O Counters: {initial_counters}")

                current_counters = proc.io_counters()  # 현재 I/O 카운터 읽기
                sent_bytes = current_counters.write_bytes - initial_counters.write_bytes
                recv_bytes = current_counters.read_bytes - initial_counters.read_bytes
                before_sent=sent_bytes
                before_Recceive = recv_bytes


                while True:
                    current_time = datetime.now()
                    current_counters = proc.io_counters()
                    sent_bytes = current_counters.write_bytes - initial_counters.write_bytes
                    recv_bytes = current_counters.read_bytes - initial_counters.read_bytes

                    if sent_bytes > before_sent:
                        print(f"Sent: {sent_bytes} bytes, Received: {recv_bytes} bytes")
                        current_time = datetime.now()
                        print("Trigger_time:", current_time)


                        if before_Recceive == recv_bytes:
                            print("Signal")
                            message = "T"
                            arduino.write(message.encode())

                        before_sent = sent_bytes
                        before_Recceive = recv_bytes
                        time.sleep(interval)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("Process terminated or access denied.")
                break


monitor_data_transfer("TDS-7130.exe", interval_ms=100)