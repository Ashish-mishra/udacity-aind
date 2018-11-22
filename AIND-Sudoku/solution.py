
# Artificial Intelligence Nanodegree - Introductory Project: Diagonal Sudoku Solver

assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    return [s+t for s in A for t in B]

boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

diagonals = [[a[0]+a[1] for a in zip(rows, cols)],[a[0]+a[1] for a in zip(rows, cols[::-1])]]
unitlist = row_units + column_units + square_units + diagonals
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[])) - set([s])) for s in boxes)

def assign_value(values, box, value):
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    candidates = [box for box in values.keys() if len(values[box]) == 2]
    nakedTwins = [[b1, b2] for b1 in candidates for b2 in peers[b1] if values[b1] == values[b2]]

    for b1, b2 in nakedTwins:
        if len(values[b1]) == 2 and len(values[b2]) == 2:
            d1 = values[b1][0]
            d2 = values[b1][1]
            common = list(set(peers[b1]) & set(peers[b2]))
            for p in common:
                if len(values[p]) > 1 and values[p] != values[b1] and values[p] != values[b2]:
                    values[p] = values[p].replace(d1, '')
                    values[p] = values[p].replace(d2, '')
    
    return values

def grid_values(grid):
    grid = ['123456789' if i=='.' else i for i in list(grid)]
    return dict(zip(boxes, grid))

def display(values):
    print(values)
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width * 3)] * 3)
    for r in rows:
        print(''.join(values[r+c].center(width) + ('|' if c in '36' else '') for c in cols))
        if r in 'CF': print(line)
    return 

def eliminate(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit, '')
    return values

def only_choice(values):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values      

def reduce_puzzle(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    values = reduce_puzzle(values)
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in boxes):
        return values
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)

    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    values = grid_values(grid)
    return search(values)

if __name__ == '__main__':
    diagonal_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diagonal_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
