"""
Contact Manager - Section III.E
Handles adding and removing contacts with smooth transitions.

Key features:
- Activation function a^[i](t) for smooth force transitions (Eq. 34, 36)
- Modified desired forces f_d^[i] = a^[i](t) f_d^[i] n^[i] (Eq. 35)
- Prevents force jumps when contacts change
"""

import numpy as np
from typing import Dict, List, Optional
import time


class ContactState:
    """Represents the state of a single contact."""
    
    def __init__(self, contact_id: int, position: np.ndarray, normal: np.ndarray):
        """
        Initialize contact state.
        
        Args:
            contact_id: Unique identifier for this contact
            position: Contact position in world frame (3D)
            normal: Contact normal vector (3D)
        """
        self.contact_id = contact_id
        self.position = position.copy()
        self.normal = normal.copy()
        self.is_active = False
        self.is_transitioning = False
        self.transition_start_time = None
        self.activation_value = 0.0
        self.mode = None  # 'adding' or 'removing'
        
    def __repr__(self):
        return (f"Contact(id={self.contact_id}, active={self.is_active}, "
                f"transition={self.is_transitioning}, a={self.activation_value:.3f})")


class ContactManager:
    """
    Manages dynamic contact changes during manipulation.
    
    Implements Section III.E:
    - Smooth contact addition (Equation 34-35)
    - Smooth contact removal (Equation 36)
    - Prevents force jumps in internal force computation
    """
    
    def __init__(self, transition_time: float = 0.1):
        """
        Initialize contact manager.
        
        Args:
            transition_time: t_a - time for smooth transition (default: 0.1s)
        """
        self.t_a = transition_time
        self.contacts: Dict[int, ContactState] = {}
        self.next_contact_id = 0
        
    def add_contact(
        self,
        position: np.ndarray,
        normal: np.ndarray,
        contact_id: Optional[int] = None
    ) -> int:
        """
        Add a new contact with smooth transition.
        
        The contact starts with activation a^[i] = 0 and ramps to 1
        over time t_a (Equation 34).
        
        Args:
            position: Contact position (3D)
            normal: Contact normal vector (3D)
            contact_id: Optional contact ID (auto-assigned if None)
            
        Returns:
            contact_id: ID of the added contact
        """
        if contact_id is None:
            contact_id = self.next_contact_id
            self.next_contact_id += 1
            
        # Create contact state
        contact = ContactState(contact_id, position, normal)
        contact.is_active = True
        contact.is_transitioning = True
        contact.transition_start_time = time.time()
        contact.mode = 'adding'
        contact.activation_value = 0.0
        
        self.contacts[contact_id] = contact
        
        return contact_id
    
    def remove_contact(self, contact_id: int):
        """
        Mark contact for removal with smooth transition.
        
        The contact activation ramps from 1 to 0 over time t_a (Equation 36).
        
        Args:
            contact_id: ID of contact to remove
        """
        if contact_id not in self.contacts:
            return
            
        contact = self.contacts[contact_id]
        contact.is_transitioning = True
        contact.transition_start_time = time.time()
        contact.mode = 'removing'
        contact.activation_value = 1.0  # Start from full activation
        
    def update_activations(self):
        """
        Update activation values for all transitioning contacts.
        
        Implements Equations 34 and 36.
        """
        current_time = time.time()
        contacts_to_delete = []
        
        for contact_id, contact in self.contacts.items():
            if not contact.is_transitioning:
                continue
                
            # Time since transition started
            t = current_time - contact.transition_start_time
            
            if contact.mode == 'adding':
                # Equation (34): a^[i](t) = t/t_a for t ≤ t_a, else 1
                if t <= self.t_a:
                    contact.activation_value = t / self.t_a
                else:
                    contact.activation_value = 1.0
                    contact.is_transitioning = False
                    
            elif contact.mode == 'removing':
                # Equation (36): a^[i](t) = 1 - t/t_a for t ≤ t_a, else 0
                if t <= self.t_a:
                    contact.activation_value = 1.0 - (t / self.t_a)
                else:
                    contact.activation_value = 0.0
                    contact.is_transitioning = False
                    contact.is_active = False
                    contacts_to_delete.append(contact_id)
                    
        # Remove fully deactivated contacts
        for contact_id in contacts_to_delete:
            del self.contacts[contact_id]
            
    def get_active_contacts(self) -> List[ContactState]:
        """Get all active contacts (a^[i] > 0)."""
        return [c for c in self.contacts.values() if c.is_active]
    
    def get_contact_positions(self) -> np.ndarray:
        """
        Get contact positions as flat vector.
        
        Returns:
            c ∈ R^(3n) - contact positions
        """
        active_contacts = self.get_active_contacts()
        if len(active_contacts) == 0:
            return np.array([])
            
        positions = [c.position for c in active_contacts]
        return np.concatenate(positions)
    
    def get_contact_normals(self) -> np.ndarray:
        """
        Get contact normals as array.
        
        Returns:
            normals (n × 3) - contact normal vectors
        """
        active_contacts = self.get_active_contacts()
        if len(active_contacts) == 0:
            return np.array([]).reshape(0, 3)
            
        normals = [c.normal for c in active_contacts]
        return np.array(normals)
    
    def get_desired_forces_with_activation(
        self,
        desired_normal_forces: np.ndarray
    ) -> np.ndarray:
        """
        Compute desired forces with activation scaling (Equation 35).
        
        f(t)_d^[i] = a^[i](t) * f_d^[i] * n^[i]
        
        Args:
            desired_normal_forces: f_d^[i] for each contact (n,)
            
        Returns:
            f_d: Scaled desired forces (3n,)
        """
        active_contacts = self.get_active_contacts()
        n_contacts = len(active_contacts)
        
        assert desired_normal_forces.shape[0] == n_contacts, \
            f"Expected {n_contacts} normal forces, got {desired_normal_forces.shape[0]}"
        
        f_d = np.zeros(3 * n_contacts)
        
        for i, contact in enumerate(active_contacts):
            # Equation (35): f(t)_d^[i] = a^[i](t) * f_d^[i] * n^[i]
            activation = contact.activation_value
            normal_force = desired_normal_forces[i]
            normal_vector = contact.normal
            
            f_d[3*i:3*i+3] = activation * normal_force * normal_vector
            
        return f_d
    
    def get_activation_vector(self) -> np.ndarray:
        """
        Get activation values for all active contacts.
        
        Returns:
            a ∈ R^n - activation values
        """
        active_contacts = self.get_active_contacts()
        return np.array([c.activation_value for c in active_contacts])
    
    def update_contact_position(
        self,
        contact_id: int,
        new_position: np.ndarray,
        new_normal: Optional[np.ndarray] = None
    ):
        """
        Update contact position (e.g., from sensor data).
        
        Args:
            contact_id: Contact ID
            new_position: New contact position (3D)
            new_normal: New contact normal (3D, optional)
        """
        if contact_id not in self.contacts:
            return
            
        contact = self.contacts[contact_id]
        contact.position = new_position.copy()
        
        if new_normal is not None:
            contact.normal = new_normal.copy()
            
    def get_contact_count(self) -> int:
        """Get number of active contacts."""
        return len(self.get_active_contacts())
    
    def clear_all_contacts(self):
        """Remove all contacts immediately."""
        self.contacts.clear()
        
    def set_transition_time(self, t_a: float):
        """Update transition time."""
        self.t_a = t_a
        
    def get_transition_time(self) -> float:
        """Get transition time."""
        return self.t_a
    
    def get_contact_info(self) -> dict:
        """
        Get detailed information about all contacts.
        
        Returns:
            Dictionary with contact information
        """
        active_contacts = self.get_active_contacts()
        
        info = {
            'n_contacts': len(active_contacts),
            'n_transitioning': sum(1 for c in active_contacts if c.is_transitioning),
            'contacts': []
        }
        
        for contact in active_contacts:
            contact_info = {
                'id': contact.contact_id,
                'position': contact.position,
                'normal': contact.normal,
                'activation': contact.activation_value,
                'is_transitioning': contact.is_transitioning,
                'mode': contact.mode
            }
            info['contacts'].append(contact_info)
            
        return info
    
    def is_stable(self) -> bool:
        """
        Check if all contacts are stable (not transitioning).
        
        Returns:
            True if no contacts are transitioning
        """
        active_contacts = self.get_active_contacts()
        return all(not c.is_transitioning for c in active_contacts)