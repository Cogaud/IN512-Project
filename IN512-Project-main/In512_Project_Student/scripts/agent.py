__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2024"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *

from threading import Thread
import numpy as np
from time import sleep, time
import random


class Agent:
    def __init__(self, server_ip):
        # --- Navigation Memories ---
        self.visited_cells = set()
        self.previous_move = None
        self.last_value = 0.0

        # Stuck detection
        self.last_position = None
        self.stuck_counter = 0

        # --- Task State (Personnel) ---
        self.completed = False
        self.my_key_found = False  # True si J'AI ramassé ma clé
        self.my_key_location = None  # Position de ma clé (si connue)
        self.my_box_location = None  # Position de ma boîte (si connue)
        self.target_pos = None  # Où je veux aller maintenant

        # --- Global Synchronization State (Collectif) ---
        self.agents_with_keys = set()  # Qui a ramassé sa clé ?
        self.agents_boxes_found = set()  # Pour qui a-t-on trouvé la boîte ?
        self.go_signal = False  # Le signal final

        # Memory of ALL found items
        self.discovered_items = {}  # Dict: (x,y) -> (owner_id, item_type)

        # Scanning Strategy
        self.scan_waypoints = []
        self.known_obstacles = set()

        # Server msg container
        self.item_data_received = None

        # --- INIT NETWORK ---
        self.network = Network(server_ip=server_ip)
        self.agent_id = self.network.id
        self.running = True
        self.network.send({"header": GET_DATA})
        self.msg = {}
        env_conf = self.network.receive()
        self.nb_agent_expected = 0
        self.nb_agent_connected = 0
        self.x, self.y = env_conf["x"], env_conf["y"]
        self.current_cell_value = env_conf.get("cell_val", 0.0)
        self.w, self.h = env_conf["w"], env_conf["h"]
        self.all_agents_positions = env_conf.get("all_agents_positions", [])

        Thread(target=self.msg_cb, daemon=True).start()
        self.wait_for_connected_agent()

    def msg_cb(self):
        """ Écoute des messages """
        while self.running:
            try:
                msg = self.network.receive()
                if msg is None:
                    # Si le serveur renvoie None, c'est probablement une réponse à GET_ITEM_OWNER sur un mur
                    self.item_data_received = {"owner": None, "type": "wall"}
                    continue
                self.msg = msg
                
                if "cell_val" in msg:
                    self.current_cell_value = msg["cell_val"]
            except Exception as e:
                print(f"Agent {self.agent_id}: Erreur réseau - {e}")
                self.running = False
                break

            if msg["header"] == MOVE:
                self.x, self.y = msg["x"], msg["y"]

            elif msg["header"] == GET_NB_AGENTS:
                self.nb_agent_expected = msg["nb_agents"]

            elif msg["header"] == GET_NB_CONNECTED_AGENTS:
                self.nb_agent_connected = msg["nb_connected_agents"]

            elif msg["header"] == GET_ITEM_OWNER:
                self.item_data_received = msg

            elif msg["header"] == BROADCAST_MSG:
                # Enregistrement global des items découverts
                if "position" in msg and "owner" in msg:
                    # Convert position to tuple if it's a list (network serialization issue)
                    pos = msg["position"]
                    if isinstance(pos, list):
                        pos = tuple(pos)
                    # Stocker avec le type d'item
                    item_type = msg.get("item_type", None)
                    if item_type is None:
                        if msg.get("Msg type") == KEY_DISCOVERED:
                            item_type = KEY_TYPE
                        elif msg.get("Msg type") == BOX_DISCOVERED:
                            item_type = BOX_TYPE
                    self.discovered_items[pos] = (msg["owner"], item_type)

                # Si une BOITE est découverte
                if msg.get("Msg type") == BOX_DISCOVERED:
                    owner = msg.get("owner")
                    pos = msg.get("position")
                    if isinstance(pos, list):
                        pos = tuple(pos)
                    self.agents_boxes_found.add(owner)
                    print(f"Agent {self.agent_id}: Box for agent {owner} discovered at {pos}")
                    if owner == self.agent_id:
                        self.my_box_location = pos
                        print(f"Agent {self.agent_id}: MY BOX is at {pos}!")

                # Si une CLÉ est DÉCOUVERTE (vue mais pas forcement prise)
                elif msg.get("Msg type") == KEY_DISCOVERED:
                    owner = msg.get("owner")
                    pos = msg.get("position")
                    if isinstance(pos, list):
                        pos = tuple(pos)
                    # Si c'est ma clé, sauvegarder sa position et la cibler
                    if owner == self.agent_id and not self.my_key_found:
                        self.my_key_location = pos
                        self.target_pos = pos

                # Si une CLÉ est RAMASSÉE (Confirmation officielle)
                if msg.get("sub_type") == "KEY_COLLECTED":
                    sender = msg.get("sender")
                    self.agents_with_keys.add(sender)
                
                # GO SIGNAL reçu d'un autre agent
                if msg.get("sub_type") == "GO_TO_BOX":
                    if not self.go_signal:
                        print(f"Agent {self.agent_id}: Received GO_TO_BOX signal!")
                        self.go_signal = True
                        self.target_pos = self.my_box_location
                        if self.my_box_location is None:
                            print(f"Agent {self.agent_id}: WARNING - Received GO signal but don't know my box location!")

    def wait_for_connected_agent(self):
        self.network.send({"header": GET_NB_AGENTS})
        check_conn_agent = True
        while check_conn_agent:
            if self.nb_agent_expected == self.nb_agent_connected:
                print("Tous les agents connectés !")
                check_conn_agent = False
            sleep(0.5)

    def get_real_val(self):
        return self.current_cell_value

    def get_perceived_value(self):
        """
        Retourne la valeur de la case.
        Masque les objets déjà trouvés SAUF si c'est ma propre clé que je cherche.
        """
        real_val = self.get_real_val()

        for (ix, iy), item_data in self.discovered_items.items():
            # Extraire owner et item_type du tuple
            if isinstance(item_data, tuple):
                owner, item_type = item_data
            else:
                owner = item_data
                item_type = None
            
            if abs(self.x - ix) <= 2 and abs(self.y - iy) <= 2:
                # Si c'est une clé déjà ramassée par son propriétaire, ignorer
                if item_type == KEY_TYPE and owner in self.agents_with_keys:
                    return 0.0

                # IMPORTANT : Si c'est MA clé et que je ne l'ai pas, JE DOIS LA VOIR (ne pas masquer)
                if owner == self.agent_id and item_type == KEY_TYPE and not self.my_key_found:
                    return real_val

                # Si c'est ma boite et que c'est le moment d'y aller
                if self.my_box_location == (ix, iy) and self.go_signal:
                    return real_val

                # Sinon (objet des autres ou ma boite trop tôt), je masque pour ne pas être distrait
                return 0.0

        return real_val

    def send_move(self, direction):
        self.network.send({"header": MOVE, "direction": direction})
        sleep(0.15)

    def check_item_owner(self):
        self.item_data_received = None
        self.network.send({"header": GET_ITEM_OWNER})
        # Petite sécurité anti-boucle infinie
        timeout = 0
        while self.item_data_received is None and timeout < 200:
            sleep(0.01)
            timeout += 1
        if self.item_data_received:
            return self.item_data_received.get("owner"), self.item_data_received.get("type")
        return None, None

    def check_global_conditions(self):
        """ Vérifie si TOUT a été trouvé """
        all_keys = len(self.agents_with_keys) >= self.nb_agent_expected
        all_boxes = len(self.agents_boxes_found) >= self.nb_agent_expected

        if all_keys and all_boxes:
            if not self.go_signal:
                print(f"Agent {self.agent_id}: TOUT EST TROUVÉ ! GO AUX BOITES !")
                print(f"Agent {self.agent_id}: My box location = {self.my_box_location}")
                print(f"Agent {self.agent_id}: Keys found: {self.agents_with_keys}, Boxes found: {self.agents_boxes_found}")
                if self.my_box_location is None:
                    print(f"Agent {self.agent_id}: WARNING - Box location unknown!")
                
                # BROADCAST GO SIGNAL to all agents
                self.network.send({
                    "header": BROADCAST_MSG,
                    "sub_type": "GO_TO_BOX"
                })
            self.go_signal = True
            self.target_pos = self.my_box_location

    def _generate_grid_zones(self):
        """
        Generate grid-based zones based on number of agents.
        Returns list of zones as (min_x, max_x, min_y, max_y) tuples.
        
        For 3 agents (assuming 35x30 grid):
        - Zone 0: Top-left (0-17, 0-19) = 18x20
        - Zone 1: Top-right (18-34, 0-19) = 17x20  
        - Zone 2: Bottom (0-34, 20-29) = 35x10
        """
        zones = []
        n = self.nb_agent_expected
        
        if n == 1:
            # Single agent covers everything
            zones.append((0, self.w, 0, self.h))
            
        elif n == 2:
            # Split vertically (left/right)
            mid_x = self.w // 2
            zones.append((0, mid_x, 0, self.h))       # Left
            zones.append((mid_x, self.w, 0, self.h))  # Right
            
        elif n == 3:
            # Grid layout: 2 zones on top, 1 zone at bottom
            # Top row: 2/3 of height, Bottom row: 1/3 of height
            top_height = (self.h * 2) // 3  # ~20 if h=30
            mid_x = self.w // 2
            
            zones.append((0, mid_x, 0, top_height))           # Top-left (18x20)
            zones.append((mid_x, self.w, 0, top_height))      # Top-right (17x20)
            zones.append((0, self.w, top_height, self.h))     # Bottom (35x10)
            
        elif n == 4:
            # 2x2 grid
            mid_x = self.w // 2
            mid_y = self.h // 2
            zones.append((0, mid_x, 0, mid_y))           # Top-left
            zones.append((mid_x, self.w, 0, mid_y))      # Top-right
            zones.append((0, mid_x, mid_y, self.h))      # Bottom-left
            zones.append((mid_x, self.w, mid_y, self.h)) # Bottom-right
            
        else:
            # For 5+ agents: Create a grid with roughly equal zones
            # Calculate grid dimensions
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            
            zone_w = self.w // cols
            zone_h = self.h // rows
            
            for i in range(n):
                col = i % cols
                row = i // cols
                min_x = col * zone_w
                max_x = (col + 1) * zone_w if col < cols - 1 else self.w
                min_y = row * zone_h
                max_y = (row + 1) * zone_h if row < rows - 1 else self.h
                zones.append((min_x, max_x, min_y, max_y))
        
        return zones

    def _assign_zone_to_agent(self, zones):
        """
        Assign each agent to their closest zone based on spawn position.
        Uses agent positions to avoid conflicts - each agent gets unique zone.
        """
        if not zones:
            return None
            
        # Calculate center of each zone
        zone_centers = []
        for (min_x, max_x, min_y, max_y) in zones:
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            zone_centers.append((cx, cy))
        
        # If we have all agent positions, do smart assignment
        if self.all_agents_positions and len(self.all_agents_positions) == self.nb_agent_expected:
            # Calculate distance from each agent to each zone center
            # Then assign zones to minimize total distance (greedy approach)
            
            assigned_zones = {}  # agent_id -> zone_index
            available_zones = set(range(len(zones)))
            
            # Sort agents by their minimum distance to any zone (greedy)
            agent_zone_distances = []
            for aid, (ax, ay) in enumerate(self.all_agents_positions):
                for zid, (cx, cy) in enumerate(zone_centers):
                    dist = np.sqrt((ax - cx)**2 + (ay - cy)**2)
                    agent_zone_distances.append((dist, aid, zid))
            
            # Sort by distance (shortest first)
            agent_zone_distances.sort(key=lambda x: x[0])
            
            # Greedy assignment
            assigned_agents = set()
            for dist, aid, zid in agent_zone_distances:
                if aid not in assigned_agents and zid in available_zones:
                    assigned_zones[aid] = zid
                    assigned_agents.add(aid)
                    available_zones.remove(zid)
                    
                    if len(assigned_agents) == len(zones):
                        break
            
            # Return my assigned zone
            if self.agent_id in assigned_zones:
                return zones[assigned_zones[self.agent_id]]
        
        # Fallback: assign zone by agent_id
        zone_idx = self.agent_id % len(zones)
        return zones[zone_idx]

    def _generate_zone_waypoints(self, min_x, max_x, min_y, max_y):
        """
        Generate snake-pattern scan waypoints within a zone.
        Serpentin HORIZONTAL: avance sur une ligne, tourne, fait 5 cases d'écart (4 cases entre lignes), tourne.
        """
        # Add margins to avoid walls at boundaries
        margin = 2
        safe_min_x = min_x + margin
        safe_max_x = max(safe_min_x + 1, max_x - margin)
        safe_min_y = min_y + margin
        safe_max_y = max(safe_min_y + 1, max_y - margin)
        
        # Espacement de 5 cases entre les lignes de scan (4 cases d'écart)
        spacing = 5
        
        # Serpentin horizontal (lignes de gauche à droite, puis droite à gauche)
        row_index = 0
        for y in range(safe_min_y, safe_max_y, spacing):
            if row_index % 2 == 0:
                # Gauche -> Droite
                for x in range(safe_min_x, safe_max_x, spacing):
                    self.scan_waypoints.append((x, y))
            else:
                # Droite -> Gauche
                last_x = safe_min_x + ((safe_max_x - safe_min_x - 1) // spacing) * spacing
                for x in range(last_x, safe_min_x - 1, -spacing):
                    self.scan_waypoints.append((x, y))
            row_index += 1
        
        print(f"Agent {self.agent_id}: {len(self.scan_waypoints)} waypoints générés (spacing={spacing})")

    def run_exploration(self):
        print(f"Agent {self.agent_id} démarre l'exploration...")

        moves_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1), 5: (-1, -1), 6: (1, -1), 7: (-1, 1), 8: (1, 1)}
        inverse_move = {1: 2, 2: 1, 3: 4, 4: 3, 5: 8, 6: 7, 7: 6, 8: 5}

        # Generation Zones de Scan - Grid-based division
        if self.nb_agent_expected > 0:
            # Define zones as rectangles: (min_x, max_x, min_y, max_y)
            zones = self._generate_grid_zones()
            
            # Assign agent to closest zone based on spawn position
            my_zone = self._assign_zone_to_agent(zones)
            
            if my_zone:
                min_x, max_x, min_y, max_y = my_zone
                print(f"Agent {self.agent_id} assigned zone: x=[{min_x},{max_x}], y=[{min_y},{max_y}]")
                
                # Generate scan waypoints for this zone
                self._generate_zone_waypoints(min_x, max_x, min_y, max_y)

        while self.running and not self.completed:
            real_val = self.get_real_val()
            perceived_val = self.get_perceived_value()
            self.visited_cells.add((self.x, self.y))

            if self.last_position == (self.x, self.y):
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            self.last_position = (self.x, self.y)

            # --- ETAPE 0 : Verifier l'état global ---
            self.check_global_conditions()

            # --- ETAPE 1 : SUR UN ITEM (PRIORITÉ ABSOLUE) ---
            if real_val == 1.0:
                owner_id, item_type = self.check_item_owner()

                if owner_id is None:
                    if self.my_box_location and (self.x, self.y) == self.my_box_location:
                        continue
                    
                    # Mark current position and nearby cells as obstacles
                    self.known_obstacles.add((self.x, self.y))
                    
                    # Skip current waypoint if it's blocked by a wall or near a wall
                    while self.scan_waypoints:
                        wp = self.scan_waypoints[0]
                        # Skip if waypoint is on or adjacent to a known obstacle
                        is_near_obstacle = False
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if (wp[0] + dx, wp[1] + dy) in self.known_obstacles:
                                    is_near_obstacle = True
                                    break
                            if is_near_obstacle:
                                break
                        if is_near_obstacle or wp == (self.x, self.y):
                            self.scan_waypoints.pop(0)
                        else:
                            break
                    
                    # EMERGENCY ESCAPE: Try all directions to get away from wall
                    escaped = False
                    # First try: inverse of previous move
                    if self.previous_move and self.previous_move in inverse_move:
                        escape_move = inverse_move[self.previous_move]
                        nx, ny = self.x + moves_map[escape_move][0], self.y + moves_map[escape_move][1]
                        if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.known_obstacles:
                            self.send_move(escape_move)
                            escaped = True
                    
                    # Second try: any valid move away from obstacles
                    if not escaped:
                        valid = self.get_valid_moves(moves_map)
                        if valid:
                            # Prefer moves that go away from obstacles
                            best_move = None
                            max_dist_from_obstacles = -1
                            for m in valid:
                                nx, ny = self.x + moves_map[m][0], self.y + moves_map[m][1]
                                min_dist = float('inf')
                                for obs in self.known_obstacles:
                                    d = abs(nx - obs[0]) + abs(ny - obs[1])
                                    min_dist = min(min_dist, d)
                                if min_dist > max_dist_from_obstacles:
                                    max_dist_from_obstacles = min_dist
                                    best_move = m
                            if best_move:
                                self.previous_move = best_move
                                self.send_move(best_move)
                                escaped = True
                    
                    # Last resort: try any move including diagonal
                    if not escaped:
                        for m in [1, 2, 3, 4, 5, 6, 7, 8]:
                            nx, ny = self.x + moves_map[m][0], self.y + moves_map[m][1]
                            if 0 <= nx < self.w and 0 <= ny < self.h:
                                self.send_move(m)
                                break
                    
                    self.stuck_counter = 0  # Reset stuck counter after escape attempt
                    continue

                if owner_id is not None:
                    # 1. Broadcast la découverte
                    msg_type = KEY_DISCOVERED if item_type == KEY_TYPE else BOX_DISCOVERED
                    if (self.x, self.y) not in self.discovered_items:
                        self.network.send({
                            "header": BROADCAST_MSG,
                            "Msg type": msg_type,
                            "position": (self.x, self.y),
                            "owner": owner_id,
                            "item_type": item_type
                        })
                        self.discovered_items[(self.x, self.y)] = (owner_id, item_type)

                        if item_type == BOX_TYPE:
                            self.agents_boxes_found.add(owner_id)
                            if owner_id == self.agent_id:
                                self.my_box_location = (self.x, self.y)
                        
                        if item_type == KEY_TYPE:
                            if owner_id == self.agent_id:
                                self.my_key_location = (self.x, self.y)

                    # 2. LOGIQUE CRITIQUE DE RAMASSAGE
                    if owner_id == self.agent_id and item_type == KEY_TYPE:
                        if not self.my_key_found:
                            print(f"Agent {self.agent_id} a RAMASSÉ sa clé (Validation Locale).")
                            # UPDATE LOCAL IMMEDIAT (Ne pas attendre le réseau)
                            self.my_key_found = True
                            self.agents_with_keys.add(self.agent_id)
                            self.target_pos = None  # On a atteint la cible
                            self.last_value = 0

                            # Informer les autres
                            self.network.send({
                                "header": BROADCAST_MSG,
                                "sub_type": "KEY_COLLECTED",
                                "sender": self.agent_id
                            })

                            # Dégager de la case pour continuer l'exploration
                            self.move_randomly_away(moves_map)
                            continue

                    # 3. Fin de partie (Sur ma boite + Signal)
                    if owner_id == self.agent_id and item_type == BOX_TYPE and self.go_signal:
                        self.completed = True
                        self.network.send({"header": BROADCAST_MSG, "Msg type": COMPLETED})
                        print("Mission Terminée !")
                        break

                    # 4. CONTINUER EXPLORATION: Si c'est une boite (la mienne ou celle d'un autre)
                    #    et ce n'est pas le moment d'y aller, on dégage et on continue le scan
                    if item_type == BOX_TYPE:
                        # Dégager de la case de la boite pour pouvoir continuer l'exploration
                        self.move_randomly_away(moves_map)
                        continue
                    
                    # 5. Si c'est la clé de quelqu'un d'autre, on dégage aussi
                    if item_type == KEY_TYPE and owner_id != self.agent_id:
                        self.move_randomly_away(moves_map)
                        continue

                # Si on est sur un item mais pas d'action spéciale, on s'éloigne
                if not self.target_pos:
                    self.move_randomly_away(moves_map)
                continue

            # --- ETAPE 1.5 : EVITEMENT OBSTACLES (AURA MUR) ---
            # Si on est dans la zone d'influence d'un mur (0.35), on detecte la direction du mur
            if 0.34 <= real_val <= 0.36:
                self.known_obstacles.add((self.x, self.y))
                
                # Try to predict where the actual wall is and mark it
                # The wall is likely in the direction we were moving
                if self.previous_move and self.previous_move in moves_map:
                    dx, dy = moves_map[self.previous_move]
                    wall_x, wall_y = self.x + dx, self.y + dy
                    if 0 <= wall_x < self.w and 0 <= wall_y < self.h:
                        self.known_obstacles.add((wall_x, wall_y))
                        # Also mark adjacent cells to the predicted wall
                        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            adj_x, adj_y = wall_x + ddx, wall_y + ddy
                            if 0 <= adj_x < self.w and 0 <= adj_y < self.h:
                                # Don't mark if it's where we came from
                                if (adj_x, adj_y) != (self.x - dx, self.y - dy):
                                    self.known_obstacles.add((adj_x, adj_y))
                
                # Back off in the opposite direction
                if self.previous_move and self.previous_move in inverse_move:
                    escape_move = inverse_move[self.previous_move]
                    nx, ny = self.x + moves_map[escape_move][0], self.y + moves_map[escape_move][1]
                    if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.known_obstacles:
                        self.send_move(escape_move)
                    else:
                        self.move_randomly_away(moves_map)
                else:
                    self.move_randomly_away(moves_map)
                continue

            # --- ETAPE 2 : CIBLE DÉFINIE ---
            if self.target_pos:
                tx, ty = self.target_pos

                # Make sure target is not in obstacles (important for box location)
                if (tx, ty) in self.known_obstacles:
                    self.known_obstacles.discard((tx, ty))

                # Anti-blocage: If stuck for too long, try smarter escape
                if self.stuck_counter > 3 and (self.x, self.y) != (tx, ty):
                    if self.stuck_counter > 10:
                        # Very stuck - clear some nearby obstacles to allow new paths
                        cells_to_clear = []
                        for obs in self.known_obstacles:
                            if abs(obs[0] - self.x) <= 3 and abs(obs[1] - self.y) <= 3:
                                cells_to_clear.append(obs)
                        for cell in cells_to_clear[:5]:  # Clear up to 5 nearby obstacles
                            self.known_obstacles.discard(cell)
                        self.stuck_counter = 0
                    
                    # Try to find an alternative route around obstacles
                    valid = self.get_valid_moves(moves_map)
                    if valid:
                        # Pick a move that isn't directly blocked and makes some progress
                        best_move = None
                        best_score = float('inf')
                        for m in valid:
                            nx, ny = self.x + moves_map[m][0], self.y + moves_map[m][1]
                            # Score: distance to target + penalty for visited cells
                            dist = np.sqrt((nx - tx)**2 + (ny - ty)**2)
                            visited_penalty = 5.0 if (nx, ny) in self.visited_cells else 0
                            score = dist + visited_penalty
                            if score < best_score:
                                best_score = score
                                best_move = m
                        if best_move:
                            self.previous_move = best_move
                            self.send_move(best_move)
                    else:
                        self.move_randomly_away(moves_map)
                    continue

                if self.x == tx and self.y == ty:
                    # On est arrivé, mais real_val n'était pas 1.0 au début de boucle ?
                    # Ça peut arriver avec la latence, on attend le prochain tour.
                    self.target_pos = None
                else:
                    move = self.get_move_towards(tx, ty, moves_map)
                    if move:
                        self.previous_move = move
                        self.send_move(move)
                    else:
                        # No valid move found - try random to escape
                        self.move_randomly_away(moves_map)
                    continue

            # --- ETAPE 3 : NAVIGATION ---

            # Gradient Strict (Chasse à l'odeur)
            # Si on sent quelque chose, on ignore le scan et on cherche activement
            if perceived_val > 0:
                # Si la valeur a baissé, on a fait fausse route -> Demi-tour immédiat
                if perceived_val < self.last_value and self.previous_move in inverse_move:
                    self.send_move(inverse_move[self.previous_move])
                    # On "oublie" la last_value pour forcer un nouveau choix au prochain tour
                    self.last_value = 0
                    continue

                # Sinon (valeur monte ou stable), on continue ou on cherche meilleur voisin
                self.last_value = perceived_val

                valid = self.get_valid_moves(moves_map)
                best = [m for m in valid if
                        (self.x + moves_map[m][0], self.y + moves_map[m][1]) not in self.visited_cells]

                if best:
                    choice = random.choice(best)
                else:
                    choice = random.choice(valid)  # Repli si tout visité

                self.previous_move = choice
                self.send_move(choice)
                continue

            # Scan de Zone (Seulement si aucune odeur et aucune cible)
            if self.scan_waypoints:
                self.last_value = 0
                wp = self.scan_waypoints[0]
                if (self.x, self.y) == wp:
                    self.scan_waypoints.pop(0)
                    continue

                if self.stuck_counter > 3:
                    self.move_randomly_away(moves_map)
                    continue

                move = self.get_move_towards(wp[0], wp[1], moves_map)
                self.send_move(move)

            # Random Fallback
            else:
                self.last_value = 0
                self.move_randomly_away(moves_map)

    def move_randomly_away(self, moves_map): # MATCHED
        valid = self.get_valid_moves(moves_map)
        if valid:
            m = random.choice(valid)
            self.previous_move = m
            self.send_move(m)

    def get_valid_moves(self, moves_map):
        """Get valid moves that don't lead to walls or out of bounds"""
        possible = []
        # Diagonal moves and their component directions
        # Move 5 (-1,-1) requires both move 1 (-1,0) and move 3 (0,-1) to be clear
        # Move 6 (1,-1) requires both move 2 (1,0) and move 3 (0,-1) to be clear
        # Move 7 (-1,1) requires both move 1 (-1,0) and move 4 (0,1) to be clear
        # Move 8 (1,1) requires both move 2 (1,0) and move 4 (0,1) to be clear
        diagonal_components = {
            5: [(-1, 0), (0, -1)],  # up-left requires up and left clear
            6: [(1, 0), (0, -1)],   # up-right requires up and right clear
            7: [(-1, 0), (0, 1)],   # down-left requires down and left clear
            8: [(1, 0), (0, 1)]     # down-right requires down and right clear
        }
        
        for m_id, (dx, dy) in moves_map.items():
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.known_obstacles:
                # For diagonal moves, check that component directions are also clear
                if m_id in diagonal_components:
                    components_clear = True
                    for cdx, cdy in diagonal_components[m_id]:
                        cx, cy = self.x + cdx, self.y + cdy
                        if (cx, cy) in self.known_obstacles:
                            components_clear = False
                            break
                    if not components_clear:
                        continue  # Skip this diagonal move - would clip through wall
                
                # Also check if we'd be moving into a corner surrounded by obstacles
                obstacle_neighbors = 0
                for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (nx + ddx, ny + ddy) in self.known_obstacles:
                        obstacle_neighbors += 1
                # Only add if not too many obstacle neighbors (avoid getting trapped)
                if obstacle_neighbors < 3:
                    possible.append(m_id)
        
        # Fallback: if no safe moves, allow any move that's not an obstacle
        if not possible:
            for m_id, (dx, dy) in moves_map.items():
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.known_obstacles:
                    possible.append(m_id)
        
        return possible

    def get_move_towards(self, tx, ty, moves_map):
        """Move towards target, preferring unvisited cells and avoiding obstacles"""
        valid_moves = self.get_valid_moves(moves_map)
        if not valid_moves:
            return None
            
        # Score each move: lower is better
        move_scores = []
        for m in valid_moves:
            nx, ny = self.x + moves_map[m][0], self.y + moves_map[m][1]
            dist = np.sqrt((nx - tx) ** 2 + (ny - ty) ** 2)
            
            # Penalize visited cells slightly to encourage exploration
            visited_penalty = 2.0 if (nx, ny) in self.visited_cells else 0.0
            
            # Heavily penalize cells adjacent to known obstacles (walls)
            obstacle_penalty = 0.0
            for (dx, dy) in moves_map.values():
                adj_x, adj_y = nx + dx, ny + dy
                if (adj_x, adj_y) in self.known_obstacles:
                    obstacle_penalty += 0.5
            
            total_score = dist + visited_penalty + obstacle_penalty
            move_scores.append((total_score, m))
        
        # Sort by score (lowest first) and return best move
        move_scores.sort(key=lambda x: x[0])
        return move_scores[0][1] if move_scores else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()
    agent = Agent(args.server_ip)
    try:
        agent.run_exploration()
    except KeyboardInterrupt:
        pass
