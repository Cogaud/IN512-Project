__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2024"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *

from threading import Thread, Lock
import numpy as np
from time import sleep
from collections import deque
import sys


class Agent:
    """ Multi-agent exploration with improved coordination for 3+ agents """
    def __init__(self, server_ip):
        # Exploration state
        self.visited = set()
        self.exploration_map = {}
        self.current_cell_val = 0
        self.blocked_cells = set()
        self.item_cells = set()
        
        # Track other agents' positions
        self.other_agents_positions = set()
        
        # Previous position for backtracking
        self.prev_x, self.prev_y = 0, 0
        self.previous_move = None
        self.last_value = 0.0
        
        # Item discovery
        self.my_key_collected = False
        self.my_key_location = None
        self.my_box_location = None
        self.all_keys = {}
        self.all_boxes = {}
        
        # Coordination
        self.completed = False
        
        # Lane exploration - FIXED for 3+ agents
        self.my_zone_start = 0
        self.my_zone_end = 0
        self.lane_spacing = 3  # Reduced from 4 for better coverage
        self.current_lane = 0
        self.lane_direction = 1
        
        # Thread safety
        self.lock = Lock()
        
        # Directions
        self.directions = {
            STAND: (0, 0),
            LEFT: (-1, 0),
            RIGHT: (1, 0),
            UP: (0, -1),
            DOWN: (0, 1),
            UP_LEFT: (-1, -1),
            UP_RIGHT: (1, -1),
            DOWN_LEFT: (-1, 1),
            DOWN_RIGHT: (1, 1)
        }
        
        # Inverse directions for gradient backtracking
        self.inverse_directions = {
            LEFT: RIGHT,
            RIGHT: LEFT,
            UP: DOWN,
            DOWN: UP,
            UP_LEFT: DOWN_RIGHT,
            UP_RIGHT: DOWN_LEFT,
            DOWN_LEFT: UP_RIGHT,
            DOWN_RIGHT: UP_LEFT
        }
        
        # Network setup
        self.network = Network(server_ip=server_ip)
        self.agent_id = self.network.id
        self.running = True
        self.network.send({"header": GET_DATA})
        self.msg = {}
        env_conf = self.network.receive()
        self.nb_agent_expected = 0
        self.nb_agent_connected = 0
        self.x, self.y = env_conf["x"], env_conf["y"]
        self.w, self.h = env_conf["w"], env_conf["h"]
        cell_val = env_conf["cell_val"]
        
        # Update other agents positions from initial data
        if "all_agents_positions" in env_conf:
            for i, (ax, ay) in enumerate(env_conf["all_agents_positions"]):
                if i != self.agent_id:
                    self.other_agents_positions.add((ax, ay))
        
        self.prev_x, self.prev_y = self.x, self.y
        self.visited.add((self.x, self.y))
        self.exploration_map[(self.x, self.y)] = cell_val
        self.current_cell_val = cell_val
        
        print(f"Agent {self.agent_id} at ({self.x}, {self.y})")
        
        Thread(target=self.msg_cb, daemon=True).start()
        self.wait_for_connected_agent()
        self.setup_zones()


    def block_wall_zone(self, x, y):
        """Block a wall cell to prevent entering it"""
        pos = (x, y)
        if pos not in self.item_cells:
            self.blocked_cells.add(pos)


    def on_wall_aura(self, x, y, cell_val):
        """Called when we step on a 0.35 cell (wall aura)"""
        if (x, y) not in self.item_cells:
            self.blocked_cells.add((x, y))
        
        if self.previous_move and self.previous_move in self.directions:
            dx, dy = self.directions[self.previous_move]
            if dx != 0 or dy != 0:
                wall_x, wall_y = x + dx, y + dy
                if 0 <= wall_x < self.w and 0 <= wall_y < self.h:
                    if (wall_x, wall_y) not in self.item_cells:
                        self.blocked_cells.add((wall_x, wall_y))


    def msg_cb(self):
        while self.running:
            try:
                msg = self.network.receive()
                if msg is None:
                    continue
                    
                with self.lock:
                    self.msg = msg
                    header = msg.get("header")
                    
                    if header == MOVE:
                        self.prev_x, self.prev_y = self.x, self.y
                        
                        self.x, self.y = msg["x"], msg["y"]
                        self.current_cell_val = msg.get("cell_val", 0)
                        pos = (self.x, self.y)
                        self.visited.add(pos)
                        self.exploration_map[pos] = self.current_cell_val
                        
                        if abs(self.current_cell_val - 0.35) < 0.02:
                            self.on_wall_aura(self.x, self.y, self.current_cell_val)
                        
                    elif header == GET_DATA:
                        # Update other agents positions
                        if "all_agents_positions" in msg:
                            self.other_agents_positions.clear()
                            for i, (ax, ay) in enumerate(msg["all_agents_positions"]):
                                if i != self.agent_id:
                                    self.other_agents_positions.add((ax, ay))
                        
                    elif header == GET_NB_AGENTS:
                        self.nb_agent_expected = msg["nb_agents"]
                        
                    elif header == GET_NB_CONNECTED_AGENTS:
                        self.nb_agent_connected = msg["nb_connected_agents"]
                        
                    elif header == GET_ITEM_OWNER:
                        self.handle_item_owner(msg)
                        
                    elif header == BROADCAST_MSG:
                        self.handle_broadcast(msg)
                        
            except Exception as e:
                pass


    def wait_for_connected_agent(self):
        self.network.send({"header": GET_NB_AGENTS})
        while True:
            sleep(0.1)
            self.network.send({"header": GET_NB_CONNECTED_AGENTS})
            sleep(0.1)
            with self.lock:
                if self.nb_agent_expected > 0 and self.nb_agent_expected == self.nb_agent_connected:
                    print(f"Agent {self.agent_id}: All connected!")
                    break
    
    
    def setup_zones(self):
        """FIXED: Better zone division for 3+ agents"""
        # Each agent gets an equal vertical slice
        zone_height = self.h // self.nb_agent_expected
        
        # Calculate zone boundaries
        self.my_zone_start = self.agent_id * zone_height
        
        # Last agent takes the remainder
        if self.agent_id == self.nb_agent_expected - 1:
            self.my_zone_end = self.h - 1
        else:
            self.my_zone_end = (self.agent_id + 1) * zone_height - 1
        
        # Start in the middle of our zone for better initial spread
        self.current_lane = self.my_zone_start + zone_height // 2
        
        # Alternate starting direction based on agent ID
        self.lane_direction = 1 if self.agent_id % 2 == 0 else -1
        
        print(f"Agent {self.agent_id}: Zone Y=[{self.my_zone_start}, {self.my_zone_end}], Start lane={self.current_lane}, Dir={self.lane_direction}")
    
    
    def is_blocked(self, pos):
        """Check if position is blocked (walls, obstacles, or other agents)"""
        x, y = pos
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return True
        
        if pos in self.item_cells:
            return False
        
        # FIXED: Consider other agents as temporary obstacles
        if pos in self.other_agents_positions:
            return True
        
        if pos in self.blocked_cells:
            return True
        
        cell_val = self.exploration_map.get(pos, None)
        if cell_val is not None and abs(cell_val - 0.35) < 0.02:
            self.block_wall_zone(pos[0], pos[1])
            return True
        
        return False
    
    
    def is_on_item_aura(self):
        """Check if we're on an undiscovered item aura"""
        val = self.current_cell_val
        
        if val < 0.2 or abs(val - 0.35) < 0.03:
            return False
        
        if abs(val - 1.0) < 0.01:
            return False
        
        if 0.2 <= val <= 0.7:
            current_pos = (self.x, self.y)
            
            all_known_items = list(self.all_keys.values()) + list(self.all_boxes.values())
            for item_pos in all_known_items:
                dist = abs(current_pos[0] - item_pos[0]) + abs(current_pos[1] - item_pos[1])
                if dist <= 3:
                    return False
            
            return True
        
        return False
    
    
    def direction_to(self, target):
        dx = target[0] - self.x
        dy = target[1] - self.y
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        for d, (ddx, ddy) in self.directions.items():
            if ddx == dx and ddy == dy:
                return d
        return STAND
    
    
    def get_perceived_value(self):
        """Returns cell value but masks already-discovered items"""
        val = self.current_cell_val
        current_pos = (self.x, self.y)
        
        all_known_items = {}
        for owner, pos in self.all_keys.items():
            all_known_items[pos] = ('key', owner)
        for owner, pos in self.all_boxes.items():
            all_known_items[pos] = ('box', owner)
        
        for item_pos, (item_type, owner) in all_known_items.items():
            if abs(self.x - item_pos[0]) <= 2 and abs(self.y - item_pos[1]) <= 2:
                if item_type == 'key' and owner == self.agent_id and not self.my_key_collected:
                    return val
                if item_type == 'box' and owner == self.agent_id and self.ready_for_box():
                    return val
                return 0.0
        
        return val
    
    
    def get_valid_exploration_moves(self):
        """Get valid moves avoiding walls, obstacles, and other agents"""
        valid_moves = []
        for d, (dx, dy) in self.directions.items():
            if d == STAND:
                continue
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                if not self.is_blocked((nx, ny)):
                    cell_val = self.exploration_map.get((nx, ny), None)
                    if cell_val is not None and abs(cell_val - 0.35) < 0.03:
                        self.block_wall_zone(nx, ny)
                        continue
                    
                    # Check diagonal clipping
                    if abs(dx) == 1 and abs(dy) == 1:
                        side1 = (self.x + dx, self.y)
                        side2 = (self.x, self.y + dy)
                        if self.is_blocked(side1) or self.is_blocked(side2):
                            continue
                    
                    valid_moves.append(d)
        return valid_moves
    
    
    def explore_lane(self):
        """IMPROVED: Lane exploration with better gradient following and agent avoidance"""
        import random
        
        # Update other agents positions MORE FREQUENTLY during navigation
        self.network.send({"header": GET_DATA})
        sleep(0.03)  # Reduced from 0.05 for faster updates
        
        current_pos = (self.x, self.y)
        
        # CRITICAL: If stuck on a wall or in blocked cell, escape immediately!
        if abs(self.current_cell_val - 0.35) < 0.02 or current_pos in self.blocked_cells:
            self.last_value = 0
            return self.escape_from_trap()
        
        perceived_val = self.get_perceived_value()
        
        # GRADIENT FOLLOWING with anti-collision and deadlock prevention
        if perceived_val > 0 and abs(perceived_val - 0.35) > 0.03:
            valid_moves = self.get_valid_exploration_moves()
            
            # If value dropped significantly, we went the wrong way - backtrack!
            if perceived_val < self.last_value - 0.05 and self.previous_move in self.inverse_directions:
                inverse_dir = self.inverse_directions[self.previous_move]
                if inverse_dir in valid_moves:
                    self.last_value = 0
                    self.previous_move = inverse_dir
                    return inverse_dir
            
            self.last_value = perceived_val
            
            if valid_moves:
                # Smart move selection: prioritize unexplored cells that avoid other agents
                unvisited_safe = []
                unvisited_any = []
                visited_safe = []
                
                for m in valid_moves:
                    nx, ny = self.x + self.directions[m][0], self.y + self.directions[m][1]
                    is_visited = (nx, ny) in self.visited
                    has_agent = (nx, ny) in self.other_agents_positions
                    
                    if not is_visited:
                        if not has_agent:
                            unvisited_safe.append(m)
                        else:
                            unvisited_any.append(m)
                    else:
                        if not has_agent:
                            visited_safe.append(m)
                
                # Priority: unvisited + no agent > unvisited + agent > visited + no agent
                if unvisited_safe:
                    choice = random.choice(unvisited_safe)
                elif visited_safe:
                    choice = random.choice(visited_safe)
                elif unvisited_any:
                    # Only go toward another agent if we're VERY close to item (val > 0.5)
                    if perceived_val > 0.5:
                        choice = random.choice(unvisited_any)
                    else:
                        # Too risky, do normal exploration instead
                        self.last_value = 0
                        return self.explore_lane_fallback()
                else:
                    choice = random.choice(valid_moves)
                
                self.previous_move = choice
                return choice
        
        # No gradient - do normal lane exploration
        return self.explore_lane_fallback()
    
    
    def explore_lane_fallback(self):
        """Normal lane-based exploration without gradient following"""
        import random
        self.last_value = 0
        
        # Navigate to current lane if not there
        if self.y != self.current_lane:
            d = self.navigate_to((self.x, self.current_lane))
            if d != STAND:
                self.previous_move = d
                return d
            # Reached lane edge, move to next lane
            self.current_lane += self.lane_spacing
            if self.current_lane > self.my_zone_end:
                self.current_lane = self.my_zone_start + 1
        
        # Move horizontally along lane
        next_x = self.x + self.lane_direction
        
        # Check if next position is blocked or occupied by another agent
        if (next_x < 0 or next_x >= self.w or 
            self.is_blocked((next_x, self.y)) or
            (next_x, self.y) in self.other_agents_positions):
            # Reverse direction and move to next lane
            self.lane_direction *= -1
            self.current_lane += self.lane_spacing
            if self.current_lane > self.my_zone_end:
                # Wrap around within our zone
                self.current_lane = self.my_zone_start + 1
                if self.current_lane > self.my_zone_end:
                    self.current_lane = self.my_zone_start
            d = self.navigate_to((self.x, self.current_lane))
            self.previous_move = d
            return d
        
        d = self.direction_to((next_x, self.y))
        self.previous_move = d
        return d
    
    
    def escape_from_trap(self):
        """Emergency escape using BFS with cardinal directions only"""
        queue = deque([(self.x, self.y, [])])
        seen = {(self.x, self.y)}
        
        while queue:
            cx, cy, path = queue.popleft()
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    
                    cell_val = self.exploration_map.get((nx, ny), None)
                    is_wall = cell_val is not None and (abs(cell_val - 0.35) < 0.02 or abs(cell_val - 1.0) < 0.01)
                    
                    # Found safe cell
                    if ((nx, ny) not in self.blocked_cells and 
                        not is_wall and 
                        (nx, ny) not in self.other_agents_positions):
                        full_path = path + [(nx, ny)]
                        if full_path:
                            return self.direction_to(full_path[0])
                    
                    queue.append((nx, ny, path + [(nx, ny)]))
        
        # Last resort
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                if (nx, ny) not in self.other_agents_positions:
                    return self.direction_to((nx, ny))
        
        return STAND
    
    
    def navigate_to(self, target):
        """FIXED: Navigate to target, allowing exploration through safe/unexplored areas"""
        if not target or (self.x, self.y) == target:
            return STAND
        
        # Phase 1: Try to find path using explored OR unexplored cells (but avoiding known obstacles)
        queue = deque([(self.x, self.y, [])])
        seen = {(self.x, self.y)}
        max_iterations = self.w * self.h
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            cx, cy, path = queue.popleft()
            
            # Try all 8 directions
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]:
                nx, ny = cx + dx, cy + dy
                
                # Out of bounds
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                
                # Already visited in this BFS
                if (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                
                # Reached target!
                if (nx, ny) == target:
                    full_path = path + [(nx, ny)]
                    if full_path:
                        return self.direction_to(full_path[0])
                    return STAND
                
                # Get cell value (None if unexplored)
                cell_val = self.exploration_map.get((nx, ny), None)
                
                # ALLOW unexplored cells when navigating to broadcast locations
                # Only SKIP cells that we KNOW are obstacles
                if cell_val is not None:
                    # Check if it's a wall (0.35 aura or confirmed blocked)
                    if abs(cell_val - 0.35) < 0.03:
                        self.blocked_cells.add((nx, ny))
                        continue
                    
                    if (nx, ny) in self.blocked_cells and (nx, ny) not in self.item_cells:
                        continue
                
                # Skip if another agent is there
                if (nx, ny) in self.other_agents_positions:
                    continue
                
                # Check diagonal movement doesn't clip through walls
                if abs(dx) == 1 and abs(dy) == 1:
                    side1 = (cx + dx, cy)
                    side2 = (cx, cy + dy)
                    
                    # Both sides must be safe
                    side1_val = self.exploration_map.get(side1, None)
                    side2_val = self.exploration_map.get(side2, None)
                    
                    # Only block diagonal if we KNOW the sides are walls
                    if side1_val is not None and abs(side1_val - 0.35) < 0.03:
                        continue
                    if side2_val is not None and abs(side2_val - 0.35) < 0.03:
                        continue
                    
                    if self.is_blocked(side1) or self.is_blocked(side2):
                        continue
                
                # Safe to add to queue (either explored-safe or unexplored)
                queue.append((nx, ny, path + [(nx, ny)]))
        
        # Phase 2: No safe path found - use greedy exploration toward target
        # This will make agent explore safely to discover path
        valid_moves = self.get_valid_exploration_moves()
        
        if valid_moves:
            target_x, target_y = target
            best_move = None
            best_dist = float('inf')
            
            for move in valid_moves:
                dx, dy = self.directions[move]
                nx, ny = self.x + dx, self.y + dy
                
                # Skip if another agent is there
                if (nx, ny) in self.other_agents_positions:
                    continue
                
                # Prefer unexplored cells when no path exists
                # This helps discover new routes
                cell_val = self.exploration_map.get((nx, ny), None)
                
                # Calculate distance to target
                dist = abs(nx - target_x) + abs(ny - target_y)
                
                # Bonus for unexplored cells (they might reveal a path)
                if cell_val is None:
                    dist -= 0.5
                
                if dist < best_dist:
                    best_dist = dist
                    best_move = move
            
            if best_move:
                return best_move
        
        # Phase 3: Completely stuck - emergency escape
        return self.escape_from_trap()
    
    
    def check_cell(self):
        with self.lock:
            cell_val = self.current_cell_val
        if abs(cell_val - 1.0) < 0.01:
            self.network.send({"header": GET_ITEM_OWNER})
            sleep(0.15)
    
    
    def handle_item_owner(self, msg):
        owner = msg.get("owner")
        item_type = msg.get("type")
        pos = (self.x, self.y)
        
        if owner is None:
            if pos not in self.item_cells:
                self.blocked_cells.add(pos)
            return
        
        self.item_cells.add(pos)
        self.blocked_cells.discard(pos)
        
        if item_type == KEY_TYPE:
            self.all_keys[owner] = pos
            if owner == self.agent_id:
                self.my_key_collected = True
                self.my_key_location = pos
                print(f"Agent {self.agent_id}: *** GOT MY KEY at {pos} ***")
            self.broadcast_item(owner, item_type, pos)
                
        elif item_type == BOX_TYPE:
            self.all_boxes[owner] = pos
            if owner == self.agent_id:
                self.my_box_location = pos
                print(f"Agent {self.agent_id}: *** FOUND MY BOX at {pos} ***")
            self.broadcast_item(owner, item_type, pos)
    
    
    def broadcast_item(self, owner, item_type, pos):
        self.network.send({
            "header": BROADCAST_MSG,
            "Msg type": KEY_DISCOVERED if item_type == KEY_TYPE else BOX_DISCOVERED,
            "position": pos,
            "owner": owner
        })
    
    
    def handle_broadcast(self, msg):
        if msg is None:
            return
        msg_type = msg.get("Msg type")
        pos_data = msg.get("position")
        if pos_data is None:
            return
        pos = tuple(pos_data)
        owner = msg.get("owner")
        
        self.item_cells.add(pos)
        self.blocked_cells.discard(pos)
        
        if msg_type == KEY_DISCOVERED:
            self.all_keys[owner] = pos
            if owner == self.agent_id and not self.my_key_location:
                self.my_key_location = pos
                print(f"Agent {self.agent_id}: Key location broadcast: {pos}")
                
        elif msg_type == BOX_DISCOVERED:
            self.all_boxes[owner] = pos
            if owner == self.agent_id and not self.my_box_location:
                self.my_box_location = pos
                print(f"Agent {self.agent_id}: Box location broadcast: {pos}")
                
        elif msg_type == COMPLETED:
            pass
    
    
    def ready_for_box(self):
        return (self.my_key_collected and 
                self.my_box_location and
                len(self.all_keys) >= self.nb_agent_expected and 
                len(self.all_boxes) >= self.nb_agent_expected)
    
    
    def need_key(self):
        return self.my_key_location and not self.my_key_collected
    
    
    def run(self):
        print(f"Agent {self.agent_id}: Starting exploration...")
        
        while not self.completed and self.running:
            try:
                self.check_cell()
                
                with self.lock:
                    pos = (self.x, self.y)
                
                # Priority 1: Get my key
                if self.need_key():
                    if pos == self.my_key_location:
                        self.check_cell()
                        sleep(0.2)
                    else:
                        # Update positions more frequently when navigating to key
                        self.network.send({"header": GET_DATA})
                        sleep(0.05)
                        d = self.navigate_to(self.my_key_location)
                        self.network.send({"header": MOVE, "direction": d})
                
                # Priority 2: Go to my box when ready
                elif self.ready_for_box():
                    if pos == self.my_box_location:
                        self.completed = True
                        self.network.send({
                            "header": BROADCAST_MSG,
                            "Msg type": COMPLETED,
                            "position": pos,
                            "owner": self.agent_id
                        })
                        print(f"Agent {self.agent_id}: *** MISSION COMPLETE! *** Visited {len(self.visited)} cells")
                        break
                    else:
                        # Update positions more frequently when navigating to box
                        self.network.send({"header": GET_DATA})
                        sleep(0.05)
                        d = self.navigate_to(self.my_box_location)
                        self.network.send({"header": MOVE, "direction": d})
                
                # Priority 3: Explore
                else:
                    d = self.explore_lane()
                    self.network.send({"header": MOVE, "direction": d})
                
                sleep(0.08)  # Slightly faster for better responsiveness
                
            except Exception as e:
                sleep(0.1)
        
        print(f"Agent {self.agent_id}: Done!")
        self.running = False
        sys.exit(0)

             
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Server IP", type=str, default="localhost")
    args = parser.parse_args()

    agent = Agent(args.server_ip)
    
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        agent.running = False
        sys.exit(0)