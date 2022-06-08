import chartalist

'''
    The following script shows an example of how to parse through the data in the 
    Stablecoin ERC20 Transactions dataset to obtain some basic statistical information.
'''

def main():
    print("Started\n")

    # Retrieve dataset by call to dataloader
    # Ethereum Stable Coin ERC20
    ethereumStableCoinERC20 = chartalist.get_dataset(dataset='ethereum',
                                                     version=chartalist.EthereumLoader.STABLECOIN_ERC20, download=True,
                                                     data_frame=True)

    # Top 5 stablecoins plus WLUNA
    stablecoin = {
        "tether": ("0xdac17f958d2ee523a2206206994597c13d831ec7".lower()),
        "usdc": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".lower()),
        "dai": ("0x6b175474e89094c44da98b954eedeac495271d0f".lower()),
        "terrausd": ("0xa47c8bf37f92aBed4A126BDA807A7b7498661acD".lower()),
        "pax": ("0x8e870d67f660d95d5be530380d0ec0bd388289e1".lower()),
        "wluna": ("0xd2877702675e6cEb975b4A1dFf9fb7BAF4C91ea9".lower())
    }

    # Total number of transactions per stable coin
    tether = 0
    usdc = 0
    dai = 0
    terrausd = 0
    pax = 0
    wluna = 0

    # Track number of transactions per day
    dailyTransactions = []
    for i in range(0, 27):
        dailyTransactions.append({
            stablecoin["tether"]: 0,
            stablecoin["usdc"]: 0,
            stablecoin["dai"]: 0,
            stablecoin["terrausd"]: 0,
            stablecoin["pax"]: 0,
            stablecoin["wluna"]: 0,
        })

    # Track number of unique EoAs/Contracts
    numExternalAddresses = {
        stablecoin["tether"]: 0,
        stablecoin["usdc"]: 0,
        stablecoin["dai"]: 0,
        stablecoin["terrausd"]: 0,
        stablecoin["pax"]: 0,
        stablecoin["wluna"]: 0,
    }

    # Keep track of what EoAs/Contracts were encountered
    uniqueExternalAddresses = {}

    # Process data
    for item in ethereumStableCoinERC20.to_dict(orient="records"):
        if item["contract_address"] == stablecoin["tether"]:
            tether += 1
        elif item["contract_address"] == stablecoin["usdc"]:
            usdc += 1
        elif item["contract_address"] == stablecoin["dai"]:
            dai += 1
        elif item["contract_address"] == stablecoin["terrausd"]:
            terrausd += 1
        elif item["contract_address"] == stablecoin["pax"]:
            pax += 1
        elif item["contract_address"] == stablecoin["wluna"]:
            wluna += 1

        if item['from_address'] not in uniqueExternalAddresses:
            uniqueExternalAddresses[item['from_address']] = 1
            numExternalAddresses[str(item['contract_address'])] += 1
        else:
            uniqueExternalAddresses[item['from_address']] += 1
        if item['to_address'] not in uniqueExternalAddresses:
            uniqueExternalAddresses[item['to_address']] = 1
            numExternalAddresses[str(item['contract_address'])] += 1
        else:
            uniqueExternalAddresses[item['to_address']] += 1

        dailyTransactions[int((item['time_stamp'] - 1651104000) / 86400)][item['contract_address']] += 1

    print("\nTotal number of transactions each stablecoin")
    print("tether " + str(tether))
    print("usdc " + str(usdc))
    print("dai " + str(dai))
    print("terrausd " + str(terrausd))
    print("pax " + str(pax))
    print("wluna " + str(wluna))

    # From Thursday, April 28, 2022 to Tuesday, May 24, 2022
    print("\nNumber of transactions per day")
    dayVal = 0
    for day in dailyTransactions:
        print(str((28 + dayVal) % 30) + ": " + str(dailyTransactions[dayVal]) + " transactions")
        dayVal += 1
        print("\n")

    print("Number of unique EoAs/Contracts per stablecoin")
    print(numExternalAddresses)

    total = 0
    for key in numExternalAddresses:
        total += numExternalAddresses[key]
    print("\nTotal number of unique EoA/Contracts: " + str(total))


    print("\nFinished \n")


if __name__ == '__main__':
    main()
