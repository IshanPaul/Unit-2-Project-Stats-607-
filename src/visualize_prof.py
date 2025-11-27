import pstats

# Load the text file
stats = pstats.Stats('profile.txt')

# Sort by cumulative time
stats.sort_stats('cumulative').print_stats(20)

# Sort by internal time
stats.sort_stats('time').print_stats(20)
