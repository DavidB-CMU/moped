#!/usr/bin/env python

##############################################################################
#
# dhcp_invoker.py - Invoke dhclient if default gateway is not available
#
# Alvaro Collet
#
##############################################################################
import commands, os, sys

# -------------------------------------------------------------------------- #
def get_default_gateway():
    """ Use route -n and some parsing to obtain the default gateway.

    Usage: IP = get_default_gateway()

    Output:
        IP - String that contains the IP address of the default gateway. 
             Returns False if the default gateway cannot be found.
    """
    def_route = '0.0.0.0'
    out = commands.getoutput('route -n')
    # Parse output and look for default route
    for lines in out.splitlines():
        words = lines.split()
        if words[0] == def_route:
            # Got default route, next item is default gateway
            return words[1]
    print "Could not find default gateway"
    return False 


# -------------------------------------------------------------------------- #
def ping_host(IP, minPings=1, maxPings=4):
    """Ping a host, return True if host is alive or False if dead.
    
        host_alive = ping_host(IP, nPings, Timeout)

        Input:
            IP - String containing IP address
        Output:
            host_alive - True if successful ping, False otherwise.
            minPings{1} - minimum number of successful pings to declare 
                          success.
            maxPings{4} - maximum number of pings before giving up

    """
    if type(IP) is str:
        result = os.system('ping -c ' + str(minPings) + ' -w ' + str(maxPings) \
                           + str(' ') + IP + ' > /dev/null')
    else:
        result = 1

    # System returns 0 if successful ping, and we want a True
    return result == 0


# -------------------------------------------------------------------------- #
def main(iface='eth0', attempts=1):
    """ Main script to invoke dhcp if default gateway not available.

    Usage: out = main()
           out = main(iface = eth0, attempts = 1)
 
    Input:
        iface{eth0} - Interface to use. By default, eth0.
        attempts{1} - Number of attempts to dhclient before giving up.
                      By default, 1.
    Output:
        out - 0 if success, 1 if failure.
    """
    def_gw = get_default_gateway()
    
    if not def_gw or ping_host(def_gw) == False:
        for att in range(attempts):
            # We couldn't ping the default gateway, call dhclient
            result = os.system('sudo dhclient ' + iface + ' > /dev/null')
            if result == 0:
                return result

        return 1
    else:
        # Pinged the default gateway correctly
        return 0
        

# -------------------------------------------------------------------------- #
if __name__ == '__main__':
    # To call this script: dhcp_invoker [interface] [attempts].
    main(*sys.argv[1:]) 


