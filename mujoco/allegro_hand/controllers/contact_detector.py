"""
Contact Detector
Extracts contact information from MuJoCo simulation.
Provides contact positions, normals, and forces needed for grasp modeling.
"""

import numpy as np
import mujoco as mj
from typing import List, Dict, Optional


class ContactDetector:
    """
    Detects and tracks contacts between fingers and object.
    """
    
    def __init__(self, model: mj.MjModel, object_body_name: str = 'object'):
        """
        Initialize contact detector.
        
        Args:
            model: MuJoCo model
            object_body_name: Name of the object body
        """
        self.model = model
        self.object_body_name = object_body_name
        
        # Get object body ID
        self.object_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_body_name)
        if self.object_id < 0:
            raise ValueError(f"Object body '{object_body_name}' not found")
        
        # Finger name mapping
        self.finger_names = {
            'ff': 'index',
            'mf': 'middle',
            'rf': 'ring',
            'th': 'thumb'
        }

    def _identify_finger(self, body_name: str) -> Optional[str]:
        """
        Identify which finger a body belongs to.
        
        Args:
            body_name: Name of the body (e.g., 'ff_tip', 'mf_link2')
            
        Returns:
            Finger identifier ('index', 'middle', 'ring', 'thumb') or None
        """
        body_name_lower = body_name.lower()
        
        for prefix, finger_name in self.finger_names.items():
            if body_name_lower.startswith(prefix):
                return finger_name
    
        return None  # Unknown/palm contact
            
    def get_object_contacts(self, data: mj.MjData) -> List[Dict]:
        """
        Get all contacts involving the object and fingertips only.
        
        Args:
            data: MuJoCo data
            
        Returns:
            List of contact dictionaries with keys:
                - 'position': Contact position in world frame (3D)
                - 'normal': Contact normal vector (3D)
                - 'force': Contact force magnitude
                - 'body_id': ID of the finger body in contact
                - 'body_name': Name of the finger body
                - 'geom1', 'geom2': Geom IDs
        """
    
        # Define fingertip body names (adjust these to match your model)
        fingertip_names = {'ff_tip', 'mf_tip', 'rf_tip', 'th_tip'}
        # Dictionary to store best contact per fingertip
        best_contacts = {}
        
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # Get body IDs for both geoms
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            
            # Check if object is involved
            if body1_id == self.object_id:
                finger_body_id = body2_id
                normal_sign = 1.0
            elif body2_id == self.object_id:
                finger_body_id = body1_id
                normal_sign = -1.0
            else:
                continue  # Not an object contact

            body_name = self.model.body(finger_body_id).name

            # Filter: only keep fingertip contacts
            if body_name not in fingertip_names:
                continue
                
                # Get contact force magnitude (penetration depth as proxy)
            penetration = -contact.dist  # Negative distance = penetration
            
            # Extract contact information
            contact_info = {
                'position': contact.pos.copy(),
                'normal': contact.frame[:3].copy() * normal_sign,
                'distance': contact.dist,
                'body_id': finger_body_id,
                'body_name': body_name,
                'finger': self._identify_finger(body_name),
                'geom1': contact.geom1,
                'geom2': contact.geom2,
                'penetration': penetration
            }
            
            # Keep only the contact with maximum penetration per fingertip
            if body_name not in best_contacts or penetration > best_contacts[body_name]['penetration']:
                best_contacts[body_name] = contact_info
        
        # Return list of contacts (without penetration field)
        contacts = []
        for body_name in sorted(best_contacts.keys()):  # Sort for consistent ordering
            contact = best_contacts[body_name]
            contact.pop('penetration')  # Remove temporary field
            contacts.append(contact)
            
        return contacts
    
    def get_contact_positions_vector(self, data: mj.MjData) -> Optional[np.ndarray]:
        """
        Get contact positions as a flat vector c ∈ R^(3n).
        
        Args:
            data: MuJoCo data
            
        Returns:
            c: Contact positions vector or None if no contacts
        """
        contacts = self.get_object_contacts(data)
        
        if len(contacts) == 0:
            return None
            
        # Stack all contact positions
        c = np.concatenate([contact['position'] for contact in contacts])
        
        return c
    
    def estimate_contact_velocities(
        self,
        data: mj.MjData,
        dt: float = 0.001
    ) -> Optional[np.ndarray]:
        """
        Estimate contact velocities using finite differences.
        
        For better accuracy, use Jacobian: ċ = J q̇
        
        Args:
            data: MuJoCo data
            dt: Time step
            
        Returns:
            ċ: Contact velocities or None
        """
        # TODO: Implement using stored history or Jacobian
        # For now, return zeros
        contacts = self.get_object_contacts(data)
        
        if len(contacts) == 0:
            return None
            
        return np.zeros(3 * len(contacts))