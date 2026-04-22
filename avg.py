import re

def calculate_average_from_brackets(filename):
    # This regex looks for digits/decimals inside parentheses
    pattern = r'\(([-+]?\d*\.\d+|\d+)\)'
    
    total_sum = 0
    count = 0
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Find all matches in the current line
                matches = re.findall(pattern, line)
                
                for match in matches:
                    total_sum += float(match)
                    count += 1
                    
        if count == 0:
            print("No values found in parentheses.")
            return 0
        
        average = total_sum / count
        
        print(f"Analysis Complete for: {filename}")
        print(f"Total values found: {count}")
        print(f"Sum of values:      {total_sum:.4f}")
        print(f"Average value:      {average:.4f}")
        
        return average

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
if __name__ == "__main__":
    calculate_average_from_brackets('social_network_output.txt')