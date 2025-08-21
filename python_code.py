# Calculate the multiplication and sum of two numbers

input_variable = 10 #int(input(("please enter random number 1")))
input_variable_2 =10 #int(input(("please enter random number 2 ")))

print(input_variable+input_variable_2)
print(input_variable*input_variable_2)

# Print the Sum of a Current Number and a Previous number
for i in range(0,10):
    print( input_variable+input_variable_2, "and Previous number is ", input_variable)

print('this will be the 3rd in put that i m sharing with git ')

prices = [7, 1, 5, 3, 6, 4]

def max_profits(prices):
    if prices !=0:
        min_price = prices[0]
        max_profit = 0
        for price in prices[1:]:
            profit = price - min_price
            max_profit = max(max_profit, profit)
            min_price = min(min_price, price)
    else:
        return 0
    return max_profit
# max_sales = max_profits(prices)
print(max_profits(prices))


nums = [2, 7, 11, 15]
target = 9
def two_sum(nums, target):
    seen ={}
    for i, num in enumerate(nums):
        needed = target - num
        if needed in seen:
            return [seen[needed],i]
        seen[num] = i
        
print(two_sum([2, 7, 11, 15], 9)) 

