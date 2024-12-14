from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import setLogLevel
import time
import random

def update_link_parameters(link, bw, delay):
    print(f"Updating link to bandwidth: {bw} Mbps and delay: {delay}...")
    link.intf1.config(bw=bw, delay=delay)
    link.intf2.config(bw=bw, delay=delay)


def point_to_point_topology():
    net = Mininet(link=TCLink)

    h1 = net.addHost("h1", ip="10.0.0.1")
    server = net.addHost("server", ip="10.0.0.3") 

    net.addLink(h1, server, intfName2="server-eth0", bw=10, delay="5ms")

    net.start()

    print("Setting TCP Cubic on h1...")
    h1.cmd("sysctl -w net.ipv4.tcp_congestion_control=cubic")

    print("Starting iperf server on server...")
    server.cmd("iperf -s &") 

    for i in range(101, 201): 
        random_bw = round(random.uniform(5, 20), 2) 
        random_delay = f"{random.randint(1, 50)}ms"  

        update_link_parameters(link=net.links[0], bw=random_bw, delay=random_delay)

        pcap_file = f"cubic_traffic_{i}.pcap"
        print(f"Starting tcpdump to capture 100 packets into {pcap_file}...")
        server.cmd(f"tcpdump -i server-eth0 -c 100 -w {pcap_file} &")
        print("Generating traffic from h1 to server...")
        h1.cmd("iperf -c 10.0.0.3 -t 5") 
        time.sleep(1)  
        server.cmd("pkill -f tcpdump")  
        print(f"Captured 100 packets into {pcap_file}.")


    print("All 100 .pcap files captured successfully.")

    print("Stopping network and cleaning up...")
    server.cmd("pkill iperf")  
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    point_to_point_topology()