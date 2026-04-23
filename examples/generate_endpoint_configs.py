"""
Generate endpoint configuration files from HCCL topology, rootinfo, and route.conf.

Input files (supports optional CLI arguments for --local mode):
- hccl_rootinfo.json (contains topo_file_path reference)
- route.conf (CPU-device pair EID mappings)

Output files: ub_endpoint_npu_*.json (one file per NPU device_id)

CLI Arguments (for --local mode):
- --local, -l: Use local testing paths (default: pod06_cpu5/)
- --server, -s: Use server production paths (/etc and /lib)
- --dry-run, -n: Parse files but do not write output
- --rootinfo-path: Path to hccl_rootinfo.json (only valid with --local)
- --topo-path: Path to topology JSON file (only valid with --local)
- --route-path: Path to route.conf file (only valid with --local)

Example usage:
python generate_endpoint_configs.py --local
python generate_endpoint_configs.py --local --rootinfo-path pod06_cpu5/hccl_rootinfo.json
     --topo-path pod06_cpu5/atlas_950_1.json --route-path pod06_cpu5/route.conf
python generate_endpoint_configs.py --server
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_route_conf(route_conf_path: Path) -> Dict[int, Dict[str, str]]:
    """
    Parse route.conf file to extract device-to-EID mappings.

    Returns: {device_id: {'local_eid': '...', 'remote_eid': '...'}}
    """
    pairs = {}
    current_device_id = None
    current_pair = {}

    with open(route_conf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '_dev_id=' in line:
                # Extract device_id: pairX_dev_id=32 -> device_id=32
                parts = line.split('=')
                current_device_id = int(parts[1])
                current_pair = {'dev_id': current_device_id, 'local_eid': None, 'remote_eid': None}
                pairs[current_device_id] = current_pair
                print(f"current_device_id: {current_device_id}")
            elif '_local_eid=' in line:
                # Extract local EID: pairX_chan0_local_eid=0x...
                eid = line.split('=')[1].strip().replace('0x', '')
                if current_device_id is not None:
                    current_pair['local_eid'] = eid
            elif '_remote_eid=' in line:
                # Extract remote EID: pairX_chan0_remote_eid=0x...
                eid = line.split('=')[1].strip().replace('0x', '')
                if current_device_id is not None:
                    current_pair['remote_eid'] = eid

    return pairs


def build_eid_to_device_map(hccl_rootinfo: Dict) -> Dict[str, int]:
    """
    Build a mapping from EID to device_id using hccl_rootinfo.

    Returns: {eid: device_id}
    """
    eid_to_device = {}
    for rank in hccl_rootinfo.get('rank_list', []):
        device_id = rank['device_id']
        for level in rank.get('level_list', []):
            for addr in level.get('rank_addr_list', []):
                eid = addr['addr']
                eid_to_device[eid] = device_id
    return eid_to_device


def find_peer_eid_from_1dmesh(
    device_id: int,
    device_local_id: int,
    device_eid: str,
    device_eid_ports: List[str],
    topo_data: Dict,
    rootinfo: Dict
) -> str:
    """
    Find the connected peer's EID for a given device from 1DM mesh topology.

    For PEER2PEER (direct connections) in net_layer 0.

    Returns: peer EID string, or empty string if not found.
    """
    # Get all 1DM mesh PEER2PEER edges
    p2p_edges = [
        edge for edge in topo_data.get('edge_list', [])
        if edge.get('topo_type') == '1DMESH'
        and edge.get('link_type') == 'PEER2PEER'
        and edge.get('net_layer') == 0
    ]

    # Build EID+port to device_id mapping from rootinfo
    eid_port_to_device = {}
    for rank in rootinfo.get('rank_list', []):
        dev_id = rank['device_id']
        for level in rank.get('level_list', []):
            if level.get('net_layer') != 0:
                continue
            for addr in level.get('rank_addr_list', []):
                eid = addr['addr']
                ports = addr.get('ports', [])
                for port in ports:
                    eid_port_to_device[(eid, port)] = dev_id

    # just use x ~ x+7, e.g. 32~39
    topo_my_index = device_id


    # Find edges involving our device and identify the peer
    for edge in p2p_edges:
        local_a = edge['local_a']
        local_b = edge['local_b']
        local_a_ports = edge.get('local_a_ports', [])
        local_b_ports = edge.get('local_b_ports', [])

        # Determine which end is our device and which is the peer
        if local_a == topo_my_index:
            my_ports = local_a_ports
            peer_ports = local_b_ports
            peer_topo_index = local_b
        elif local_b == topo_my_index:
            my_ports = local_b_ports
            peer_ports = local_a_ports
            peer_topo_index = local_a
        else:
            continue

        # Find intersection of our EID's ports and the connected ports
        for my_port in device_eid_ports:
            if my_port in my_ports:
                # Find which peer device connects through this port
                # The peer port is the corresponding one on the other side
                # For ring topology there's typically a 1:1 port mapping
                for peer_port in peer_ports:
                    # Look up device_id for this topology index
                    # and find EID with this port
                    peer_device_id = None
                    for rank in rootinfo.get('rank_list', []):
                        if rank['device_id'] != peer_topo_index:
                            continue
                        for level in rank.get('level_list', []):
                            if level.get('net_layer') != 0:
                                continue
                            for addr in level.get('rank_addr_list', []):
                                if peer_port in addr.get('ports', []):
                                    return addr['addr']

    return ""


def get_protocol_from_eid(
    eid: str,
    net_layer: int,
    topo_data: Dict,
    device_local_id: int,
    device_eid_ports: List[str]
) -> str:
    """
    Determine protocol for an EID based on topology and net layer.

    Returns: 'ub_ctp' or 'ub_tp'.

    todo: 'roce', 'uboe'
    """
    # For net_layer 0 (1DMESH), always use ub_ctp
    if net_layer == 0:
        return 'ub_ctp'

    # For CLOS layer (net_layer 1+), check topology
    if net_layer >= 1:
        for edge in topo_data.get('edge_list', []):
            if edge.get('topo_type') == 'CLOS' and edge.get('net_layer') == net_layer:
                protocols = edge.get('protocols', [])
                # Check if any of the device's EID ports match this edge's ports
                edge_a_ports = edge.get('local_a_ports', [])
                edge_b_ports = edge.get('local_b_ports', [])
                all_edge_ports = set(edge_a_ports) | set(edge_b_ports)
                device_port_set = set(device_eid_ports)

                # If there's any port overlap, this edge applies to this device
                if all_edge_ports & device_port_set:
                    if 'UB_CTP' in protocols:
                        return 'ub_ctp'
                    elif 'UB_TP' in protocols:
                        return 'ub_tp'

    return 'ub_ctp'


def get_h2d_plane_id(device_id: int, rootinfo: Dict) -> str:
    """
    Find the host endpoint (with 6 ports) for a device and return its plane_id.

    Host endpoints are located at net_layer=1 (CLOS layer), on the UDIE with larger PG.

    Args:
        device_id: The NPU device ID (e.g., 32, 33, 34, etc.)
        rootinfo: Parsed hccl_rootinfo.json data

    Returns:
        The plane_id from the H2D endpoint, or 'plane_0' if not found
    """
    for rank in rootinfo.get('rank_list', []):
        if rank['device_id'] == device_id:
            for level in rank.get('level_list', []):
                if level.get('net_layer') == 1:  # CLOS layer
                    for addr_entry in level.get('rank_addr_list', []):
                        if len(addr_entry.get('ports', [])) == 6:  
                            # 1DPoD: H2D on the UDIE with PG = 6 ports
                            return addr_entry.get('plane_id', 'plane_0')
    return 'plane_0'  # Default fallback if not found


def generate_endpoint_list(
    device_id: int,
    device_info: Dict,
    topo_data: Dict,
    rootinfo: Dict
) -> List[Dict]:
    """
    Generate endpoint list for a single NPU device.
    """
    endpoint_list = []
    device_local_id = device_info['local_id']
    seen_eids = set()  # Track unique EIDs

    # Iterate through all layers in level_list
    for level in device_info['level_list']:
        net_layer = level['net_layer']
        net_type = level['net_type']

        for addr_entry in level['rank_addr_list']:
            eid = addr_entry['addr']
            device_eid_ports = addr_entry.get('ports', [])
            plane_id = addr_entry.get('plane_id', 'plane_0')

            # Skip duplicate EIDs
            if eid in seen_eids:
                continue
            seen_eids.add(eid)

            # Determine protocol
            protocol = get_protocol_from_eid(eid, net_layer, topo_data, device_local_id, device_eid_ports)

            # Create endpoint
            endpoint = {
                "protocol": protocol,
                "comm_id": eid,
                "placement": "device"
            }

            # Add plane field for ub_tp protocol or for net_layer >= 1 (CLOS/P2N)
            # P2P (net_layer 0) direct connections should NOT have plane field
            if (net_layer >= 1 or protocol == 'ub_tp') and plane_id:
                endpoint["plane"] = plane_id

            # Add dst_eid for 1DMESH (net_layer 0) direct connections
            if net_layer == 0:
                peer_eid = find_peer_eid_from_1dmesh(
                    device_id, device_local_id, eid, device_eid_ports, topo_data, rootinfo
                )
                if peer_eid:
                    endpoint["dst_eid"] = peer_eid

            endpoint_list.append(endpoint)

    return endpoint_list


if __name__ == "__main__":
    """Main entry point for endpoint config generation."""
    parser = argparse.ArgumentParser(
        description="Generate NPU endpoint configuration files from HCCL topology and route.conf."
    )
    parser.add_argument("--local", "-l", action="store_true", default=True,
                       help="Use local testing paths (pod06_cpu5 directory)")
    parser.add_argument("--server", "-s", action="store_true",
                       help="Use server production paths (/etc and /lib)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Parse files but do not write output")
    parser.add_argument("--rootinfo-path", type=str, default=None,
                       help="Path to hccl_rootinfo.json file (only valid with --local)")
    parser.add_argument("--topo-path", type=str, default=None,
                       help="Path to topology JSON file (only valid with --local)")
    parser.add_argument("--route-path", type=str, default=None,
                       help="Path to route.conf file (only valid with --local)")

    args = parser.parse_args()

    # Determine mode
    use_local = not args.server

    # Set default paths for local mode
    if use_local and not args.rootinfo_path:
        args.rootinfo_path = "D:/gitrep/newfeature/localcommres/pod06_cpu5/hccl_rootinfo.json"
        args.topo_path = "D:/gitrep/newfeature/localcommres/pod06_cpu5/atlas_950_1.json"
        args.route_path = "D:/gitrep/newfeature/localcommres/pod06_cpu5/route.conf"

    mode_str = f"local" if use_local else "server"
    print(f"Running in {mode_str} mode")
    if args.server:
        args.rootinfo_path = "/etc/hccl_rootinfo.json"
        args.route_path = "/lib/route.conf"

    if args.dry_run:
        print("Dry run mode: parsing only, no output files will be written")

    # Parse route.conf
    print(f"Loading: {args.route_path}")
    route_pairs = parse_route_conf(args.route_path)
    print(f"Found {len(route_pairs)} device pairs (device_id: {sorted(route_pairs.keys())})")

    # Load hccl_rootinfo.json
    print(f"Loading: {args.rootinfo_path}")
    with open(args.rootinfo_path) as f:
        hccl_rootinfo = json.load(f)

    if args.server:
        args.topo_path = Path(hccl_rootinfo['topo_file_path'])

    # Load topology file
    print(f"Loading topology: {args.topo_path}")
    with open(args.topo_path) as f:
        topo_data = json.load(f)

    # Generate endpoint files for each device
    print(f"Generating endpoints for {len(route_pairs)} NPUs...")
    for device_id, route_pair_info in route_pairs.items():
        # Find device info
        device_info = None
        for rank in hccl_rootinfo.get('rank_list', []):
            if rank['device_id'] == device_id:
                device_info = rank
                break

        if not device_info:
            print(f"Warning: device_id {device_id} not found in hccl_rootinfo")
            continue

        # Generate endpoint list
        endpoint_list = generate_endpoint_list(
            device_id, device_info, topo_data, hccl_rootinfo
        )

        # Add CPU host endpoint from route.conf
        if 'local_eid' in route_pair_info:
            h2d_plane_id = get_h2d_plane_id(device_id, hccl_rootinfo)
            host_endpoint = {
                "protocol": "ub_ctp",
                "comm_id": route_pair_info['local_eid'],
                "placement": "host",
                "plane": h2d_plane_id
            }
            # Insert host endpoint at beginning
            endpoint_list.insert(0, host_endpoint)

        output = {
            "version": "1.3",
            "net_instance_id": device_info['level_list'][0]['net_instance_id'],
            "endpoint_list": endpoint_list
        }

        output_path = Path(f"/etc/hixlep/ub_endpoint_npu_{device_id}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.dry_run:
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(output, f, indent=2)
            print(f"Generated: {output_path}")
        else:
            print(f"[Dry run] Would generate: {output_path}")

    print(f"{'Dry run: Would generate' if args.dry_run else 'Generated'} {len(route_pairs)} endpoint configuration files.")



