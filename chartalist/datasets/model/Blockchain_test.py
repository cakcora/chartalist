import hashlib
from datetime import date, datetime
import time
from random import random, randint, shuffle, sample


class Block:
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce  # set to zero as default not applicable in first section
        self.previous_hash = previous_hash
        self.hash = self.hash_block()

    def hash_block(self):
        # Your code for QUESTION 1 Here
        hash = hashlib.sha256(
            (str(self.index) + str(self.timestamp) + self.data + self.previous_hash + str(self.nonce)).encode(
                'utf-8')).hexdigest()
        return hash

    # Creates the first block with current time and generic data


def create_genesis_block():
    # Manually construct a block with
    # index zero and arbitrary previous hash
    return Block(0, datetime.now(), "Genesis Block", "0")

    # Function that creates the next block, given the last block on the chain you want to mine on


def next_block(last_block, nonce=0):
    # Your code for QUESTION 2 here
    block = Block(last_block.index + 1, datetime.now(), "Hey! I'm block {}".format(str(last_block.index + 1)),
                  last_block.hash)
    return block


# blockchain = [create_genesis_block()]
#
# # Create our initial reference to previous block which points to the genesis block
# previous_block = blockchain[0]
#
# # How many blocks should we add to the chain after the genesis block
# num_blocks = 20


def complete_chain(num_blocks, blockchain, previous_block):
    # Add blocks to the chain
    for i in range(0, num_blocks):
        # Your code for QUESTION 3 Here
        block_to_add = next_block(blockchain[-1])
        blockchain.append(block_to_add)
        # Your code for QUESTION 3 ends Here
        # Tell everyone about it!
        print("Block #{} has been added to the blockchain!".format(block_to_add.index))
        print("Hash: {}\n".format(block_to_add.hash))


# complete_chain(num_blocks, blockchain, previous_block)


# def test_question_3(blockchain, num_blocks):
#     correct = True
#     if len(blockchain) != num_blocks + 1:
#         correct = False
#     for i in range(len(blockchain)-1):
#         if blockchain[i + 1].previous_hash != blockchain[i].hash:
#             correct = False
#             break
#     print_statement = "PASSED!!! Move on to the next Part" if correct else "FAILED!!! Try Again :("
#     print(print_statement)
#
# test_question_3(blockchain, num_blocks)



blockchain_pow = [create_genesis_block()]

# Create our initial reference to previous block which points to the genesis block
previous_block = blockchain_pow[0]

# How many blocks should we add to the chain after genesis block
num_blocks = 20

# magnitude of difficulty of hash - number of zeroes that must be in the beginning of the hash
difficulty = 3

# length of nonce that will be generated and added
nonce_length = 20


def generate_nonce(length=20):
    return ''.join([str(randint(0, 9)) for i in range(length)])


def generate_difficulty_bound(difficulty=1):
    diff_str = ""
    for i in range(difficulty):
        diff_str += '0'
    for i in range(64 - difficulty):
        diff_str += 'F'
    diff_str = "0x" + diff_str  # "0x" needs to be added at the front to specify that it is a hex representation
    return (
        int(diff_str, 16))  # Specifies that we want to create an integer of base 16 (as opposed to the default base 10)


# Given a previous block and a difficulty metric, finds a nonce that results in a lower hash value

def find_next_block(last_block, difficulty, nonce_length):
    difficulty_bound = generate_difficulty_bound(difficulty)
    start = time.process_time()
    new_block = next_block(last_block)
    hashes_tried = 1
    # Your code for QUESTION 4 Starts here
    while int ("0x" + new_block.hash,16) > difficulty_bound:
        hashes_tried += 1
        new_block = next_block(last_block, generate_nonce(nonce_length))
    # Your code for QUESTION 4 Ends here
    time_taken = time.process_time() - start
    return (time_taken, hashes_tried, new_block)





# Add blocks to the chain based on difficulty with nonces of length nonce_length
def create_pow_blockchain(num_blocks, difficulty, blockchain_pow, previous_block, nonce_length, print_data=1):
    hash_array = []
    time_array = []
    for i in range(0, num_blocks):
        time_taken, hashes_tried, block_to_add = find_next_block(previous_block, difficulty, nonce_length)
        blockchain_pow.append(block_to_add)
        previous_block = block_to_add
        hash_array.append(hashes_tried)
        time_array.append(time_taken)
        # Tell everyone about it!
        if print_data:
            print("Block #{} has been added to the blockchain!".format(block_to_add.index))
            print("{} Hashes Tried!".format(hashes_tried))
            print("Time taken to find block: {}".format(time_taken))
            print("Hash: {}\n".format(block_to_add.hash))
    return (hash_array, time_array)


#hash_array, time_array = create_pow_blockchain(num_blocks, difficulty, blockchain_pow, previous_block, nonce_length)


def test_question_4(blockchain_pow, num_blocks):
    correct = True
    bound = generate_difficulty_bound(difficulty)
    if len(blockchain_pow) != num_blocks + 1:
        correct = False
    for i in range(len(blockchain_pow) - 1):
        if blockchain_pow[i + 1].previous_hash != blockchain_pow[i].hash:
            correct = False
            break
        if int(blockchain_pow[i + 1].hash, 16) > bound:
            correct = False
            break
    print_statement = "PASSED!!!  Move on to the next Part" if correct else "FAILED!!! Try Again :("
    print(print_statement)


#test_question_4(blockchain_pow, num_blocks)


# Naive miner class that races with other miners to see who can get a certain number of blocks first
class MinerNodeNaive:
    def __init__(self, name, compute):
        self.name = name
        self.compute = compute

    def try_hash(self, diff_value, chain):
        last_block = chain[-1]
        difficulty = generate_difficulty_bound(diff_value)
        date_now = datetime.now()
        this_index = last_block.index + 1
        this_timestamp = date_now
        this_data = "Hey! I'm block " + str(this_index)
        this_hash = last_block.hash
        new_block = Block(this_index, this_timestamp, this_data, this_hash)
        if int(new_block.hash, 16) < difficulty:
            chain.append(new_block)
            # Tell everyone about it!
            print("Block #{} has been added to the blockchain!".format(new_block.index))
            print("Block found by: {}".format(self.name))
            print("Hash: {}\n".format(new_block.hash))


#Initialize multiple miners on the network
uis_Miner = MinerNodeNaive("Stavanger Miner", 10)
uio_Miner = MinerNodeNaive("Oslo Miner", 5)
uit_Miner = MinerNodeNaive("Tromsø Miner", 2)
cornell_Miner = MinerNodeNaive("Ithaca Miner", 1)

miner_array = [uis_Miner, uio_Miner, uit_Miner, cornell_Miner]
def create_compute_simulation(miner_array):
    compute_array = []
    for miner in miner_array:
        for i in range(miner.compute):
            compute_array.append(miner.name)
    return(compute_array)

compute_simulation_array = create_compute_simulation(miner_array)
shuffle(compute_simulation_array)
chain_length = 20
blockchain_distributed = [create_genesis_block()]
genesis_block_dist = blockchain_distributed[0]
chain_difficulty = [randint(2,4) for i in range(chain_length)]
for i in range(len(chain_difficulty)):
    while len(blockchain_distributed) < i + 2:
        next_miner_str = sample(compute_simulation_array, 1)[0]
        next_miner = uis_Miner #random default (go bears)
        for miner in miner_array:
            if next_miner_str == miner.name:
                next_miner = miner
        next_miner.try_hash(chain_difficulty[i], blockchain_distributed)