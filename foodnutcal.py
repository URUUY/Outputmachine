import requests
import csv

# Assume you have a tab-separated text file with format: food_name\tpercentage
INPUT_FILE = 'result.txt'
OUTPUT_FILE = 'food_calories_output.csv'

# Nutritionix API configuration
API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"
API_KEY = "87c0202d825cd21d43e51a8fdd8015b1"
APP_ID = "c3bb6459"

def read_food_percentage(file_path):
    food_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                try:
                    food, percentage = line.strip().split('\t')
                    food_list.append((food, percentage))
                except ValueError:
                    print(f"Skipping malformed line: {line}")
    return food_list

def get_food_nutrition(food_name):
    headers = {
        'x-app-id': APP_ID,
        'x-app-key': API_KEY,
        'x-remote-user-id': '0', 
        'Content-Type': 'application/json'
    }
    payload = {"query": food_name}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status() 
        data = response.json()
        if 'foods' in data and len(data['foods']) > 0:
            food_info = data['foods'][0]
            return {
                'calories': food_info.get('nf_calories'),
                'weight_grams': food_info.get('nf_serving_weight_grams'),
                'serving_size': food_info.get('serving_weight_grams')  # Alternative field
            }
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {food_name}: {e}")
    return None

def write_output(file_path, food_data):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Food Name", "Percentage", "Calories (kcal)", "Weight (g)", "Serving Size"])
        for food, percentage, calories, weight, serving_size in food_data:
            writer.writerow([food, percentage, calories, weight, serving_size])

def main():
    food_list = read_food_percentage(INPUT_FILE)
    food_data = []
    
    for food, percentage in food_list:
        try:
            percentage_float = float(percentage.strip('%'))
            if percentage_float > 1:  # Only process foods with >1% presence
                nutrition = get_food_nutrition(food)
                if nutrition is not None:
                    calories = nutrition['calories']
                    weight = nutrition['weight_grams']
                    serving_size = nutrition['serving_size']
                    
                    # Use weight_grams first, fall back to serving_size if not available
                    final_weight = weight if weight is not None else serving_size
                    
                    food_data.append((food, percentage, calories, final_weight, serving_size))
                    print(f"Found data for {food}: {calories} kcal, {final_weight}g")
                else:
                    print(f"No nutrition data found for {food}")
        except ValueError:
            print(f"Invalid percentage value {percentage} for {food}")

    write_output(OUTPUT_FILE, food_data)
    print(f"Results successfully written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()