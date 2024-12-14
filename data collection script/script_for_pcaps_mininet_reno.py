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
    # Create the network
    net = Mininet(link=TCLink)

    # Add hosts and server
    h1 = net.addHost("h1", ip="10.0.0.1")  # Client
    server = net.addHost("server", ip="10.0.0.3")  # Server

    # Add initial link (parameters will be updated in the loop)
    net.addLink(h1, server, intfName2="server-eth0", bw=10, delay="5ms")

    # Start the network
    net.start()

    # Set TCP congestion control algorithm
    print("Setting TCP Reno on h1...")
    h1.cmd("sysctl -w net.ipv4.tcp_congestion_control=reno")

    # Start iperf server on the server
    print("Starting iperf server on server...")
    server.cmd("iperf -s &")  # Run iperf server in the background

    # Automatically capture 100 .pcap files with random link settings
    for i in range(101, 201):  # Loop for 100 captures
        # Generate random bandwidth (5 Mbps to 20 Mbps) and delay (1 ms to 50 ms)
        random_bw = round(random.uniform(5, 20), 2)  # Bandwidth in Mbps
        random_delay = f"{random.randint(1, 50)}ms"  # Delay in milliseconds

        # Update link parameters dynamically
        update_link_parameters(link=net.links[0], bw=random_bw, delay=random_delay)

        # Capture packets
        pcap_file = f"reno_traffic_{i}.pcap"
        print(f"Starting tcpdump to capture 100 packets into {pcap_file}...")
        server.cmd(f"tcpdump -i server-eth0 -c 100 -w {pcap_file} &")
        print("Generating traffic from h1 to server...")
        h1.cmd("iperf -c 10.0.0.3 -t 5")  # Generate traffic for 5 seconds
        time.sleep(1)  # Allow some buffer time for tcpdump to finish
        server.cmd("pkill -f tcpdump")  # Stop tcpdump after capture
        print(f"Captured 100 packets into {pcap_file}.")


    print("All 100 .pcap files captured successfully.")

    # Stop the network
    print("Stopping network and cleaning up...")
    server.cmd("pkill iperf")  # Stop iperf server
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    point_to_point_topology()