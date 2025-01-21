from openai import OpenAI
from typing import List, Dict
import time

def get_active_batches(client: OpenAI) -> List[Dict]:
    """
    Retrieves all active batches that can be cancelled.
    Active states are those that haven't reached a terminal state.
    """
    active_batches = []
    
    # Terminal states that can't be cancelled
    terminal_states = {'completed', 'failed', 'expired', 'cancelled'}
    
    # Paginate through all batches
    has_more = True
    after = None
    
    while has_more:
        response = client.batches.list(limit=100, after=after)
        
        # Filter for active batches
        for batch in response.data:
            if batch.status not in terminal_states:
                active_batches.append(batch)
        
        # Check if there are more pages
        has_more = response.has_more
        if has_more and response.data:
            after = response.data[-1].id
    
    return active_batches

def cancel_active_batches(client: OpenAI, dry_run: bool = True) -> Dict[str, int]:
    """
    Cancels all active batches.
    
    Args:
        client: OpenAI client instance
        dry_run: If True, only prints what would be cancelled without actually cancelling
    
    Returns:
        Dictionary with counts of batches in different states
    """
    active_batches = get_active_batches(client)
    
    stats = {
        'total_active': len(active_batches),
        'cancelled': 0,
        'errors': 0
    }
    
    if dry_run:
        print(f"Found {stats['total_active']} active batches that would be cancelled:")
        for batch in active_batches:
            print(f"  - Batch ID: {batch.id} (Status: {batch.status})")
        return stats
    
    # Actually cancel the batches
    print(f"Cancelling {stats['total_active']} active batches...")
    
    for batch in active_batches:
        try:
            print(f"Cancelling batch {batch.id} (current status: {batch.status})...")
            client.batches.cancel(batch.id)
            stats['cancelled'] += 1
            
        except Exception as e:
            print(f"Error cancelling batch {batch.id}: {str(e)}")
            stats['errors'] += 1
    
    return stats

def main():
    client = OpenAI()
    
    # First do a dry run to see what would be cancelled
    print("Performing dry run...")
    dry_run_stats = cancel_active_batches(client, dry_run=True)
    
    if dry_run_stats['total_active'] == 0:
        print("No active batches found.")
        return
    
    # Prompt for confirmation
    response = input("\nDo you want to proceed with cancellation? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\nProceeding with batch cancellation...")
        stats = cancel_active_batches(client, dry_run=False)
        
        print("\nCancellation complete!")
        print(f"Successfully cancelled: {stats['cancelled']} batches")
        if stats['errors'] > 0:
            print(f"Errors encountered: {stats['errors']} batches")
    else:
        print("\nOperation cancelled by user.")

if __name__ == "__main__":
    main()
