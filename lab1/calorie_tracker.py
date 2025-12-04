# ------------------------------------------------------
# Name: Jagrit Jaggi
# Date: November 6, 2025
# Project: Daily Calorie Tracker
# ------------------------------------------------------

print("==================================================")
print("     Welcome to Daily Calorie Tracker     ")
print("==================================================")
print("A simple way to stay mindful of what you eat — log your meals, see your total calories, and make sure you’re meeting your daily goals. Helps you get in a better shape\n")

# Task 2: Input and Data Collection
meal_name = []
meal_calories = []

num_meals = int(input("How many meals would you like to enter today? "))

for i in range(num_meals):
    meal = input(f"\nEnter meal name #{i+1}: ")
    calories = float(input(f"Enter calories for {meal}: "))
    meal_name.append(meal)
    meal_calories.append(calories)
    
# Task 3: Calorie Calculations
total_calories = sum(meal_calories)
average_calories = total_calories / len(meal_calories)

daily_limit = float(input("\nEnter your daily calorie limit: "))

print(f"\nTotal calories consumed: {total_calories}")
print(f"Average calories per meal: {average_calories:.2f}")

# Task 4: Exceed Limit Warning System
if total_calories > daily_limit:
    print("\n Warning: You have exceeded your daily calorie limit!")
else:
    print("\n Great job! You are within your daily calorie limit.")
    
# Task 5: Neatly Formatted Output
print("\n==================================================")
print("         Calorie Tracker Summary         ")
print("==================================================")
print("Meal Name\t\tCalories")
print("--------------------------------------------------")

for meal, cal in zip(meal_name, meal_calories):
    print(f"{meal:<15}\t{cal}")

print("--------------------------------------------------")
print(f"Total:\t\t\t{total_calories}")
print(f"Average:\t\t{average_calories:.2f}")

# Task 6 (Bonus): Save Session Log to File
import datetime

save_option = input("\nWould you like to save this report? (yes/no): ").lower()

if save_option == "yes":
    filename = f"calorie_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as file:
        file.write("=== Daily Calorie Tracker Report ===\n")
        file.write(f"Date: {datetime.datetime.now()}\n\n")
        file.write("Meal Name\tCalories\n")
        file.write("----------------------------------------\n")
        for meal, cal in zip(meal_name, meal_calories):
            file.write(f"{meal:<15}\t{cal}\n")
        file.write("----------------------------------------\n")
        file.write(f"Total:\t\t{total_calories}\n")
        file.write(f"Average:\t{average_calories:.2f}\n")
        file.write(f"Daily Limit:\t{daily_limit}\n")
        if total_calories > daily_limit:
            file.write("\nStatus: Exceeded limit ️\n")
        else:
            file.write("\nStatus: Within limit \n")

    print(f"\nReport saved successfully as '{filename}'")
else:
    print("\nReport not saved. Goodbye!")



print("\nGoodbye! Thank you! Visit Again...")
