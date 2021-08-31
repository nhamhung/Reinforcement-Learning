from utils import shorten_actions, convert_action_to_move
from collections import defaultdict, deque


class GNode(object):
    def __init__(self):
        self.in_degrees = 0
        self.out_nodes = []


def group_actions(actions):
    converted_actions = [convert_action_to_move(
        action, 6, index=1) for action in actions]

    shortened_actions = shorten_actions(converted_actions)

    dependencies = defaultdict(list)
    for i, action in enumerate(shortened_actions):
        for j in range(i - 1, -1, -1):
            nearest_action = shortened_actions[j]
            if list(set(action).intersection(set(nearest_action))):
                dependencies[j].append(i)
                break

    # Create a graph which is a dictionary of nodes
    graph = defaultdict(GNode)

    # Populate the graph, map each parent node to all its dependent children, compute indegrees of each dependent child
    for node, dependent_nodes in dependencies.items():
        for dependent_node in dependent_nodes:
            graph[node].out_nodes.append(dependent_node)
            graph[dependent_node].in_degrees += 1

    # Start grouping all nodes with no indegrees
    no_dependency_nodes = deque()
    for index, node in graph.items():
        if node.in_degrees == 0:
            no_dependency_nodes.append(index)

    # Initialize for storing each no dependency group
    no_dependency_nodes_count = len(no_dependency_nodes)
    all_action_groups = []
    each_action_group = []

    counter = 0
    while no_dependency_nodes:
        # Remove each no dependency node
        action = no_dependency_nodes.popleft()

        # Append that node to current group
        each_action_group.append(shortened_actions[action])

        # For all dependent children node, update their state after prev node removed and also append them to queue if they now have no dependency
        for next_action in graph[action].out_nodes:
            graph[next_action].in_degrees -= 1

            if graph[next_action].in_degrees == 0:
                no_dependency_nodes.append(next_action)

        counter += 1

        # If all nodes in current no dependency group has been popped, append this action group, reset counter, update new current no dependency group which is actually the queue
        if counter == no_dependency_nodes_count:
            all_action_groups.append(each_action_group)
            each_action_group = []
            counter = 0
            no_dependency_nodes_count = len(no_dependency_nodes)

    return all_action_groups


def get_action_groups_str(all_action_groups):
    group_str = ''

    for i, group in enumerate(all_action_groups):
        group_str += f'Group {i+1} moves: ' + \
            ' '.join(map(str, group)) + '\n'

    return group_str
