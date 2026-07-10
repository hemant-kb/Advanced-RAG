def factorial(n):
    # Factorial is only defined for non-negative integers.
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")

    # Start with 1 because factorial multiplication builds from 1 upward.
    result = 1

    # Multiply all integers from 2 to n.
    for i in range(2, n + 1):
        result *= i

    # Return the computed factorial value.
    return result

# Example: read input, compute factorial, and print result.
num = int(input("Enter a number: "))
print(f"Factorial of {num} is {factorial(num)}")
