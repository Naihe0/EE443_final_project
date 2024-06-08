def reformat_line(line):
    parts = line.split()
    if len(parts) != 11:
        raise ValueError("Each line must contain exactly 11 space-separated values")
    
    frame_id = parts[0]
    track_id = parts[2]
    xmin = parts[3]
    ymin = parts[4]
    width = parts[5]
    height = parts[6]
    
    reformatted_line = f"8,{track_id},{frame_id},{xmin},{ymin},{width},{height},-1,-1"
    return reformatted_line

def reformat_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            reformatted_line = reformat_line(line.strip())
            outfile.write(reformatted_line + '\n')

# Replace 'input.txt' and 'output.txt' with your actual file names
input_file = 'input.txt'
output_file = 'output.txt'
reformat_file(input_file, output_file)
