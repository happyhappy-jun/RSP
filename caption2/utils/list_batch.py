from openai import OpenAI
from datetime import datetime


def format_timestamp(timestamp):
    """Convert Unix timestamp to readable datetime."""
    if timestamp is None:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def display_batch_info(batch):
    """Format and display information for a single batch."""
    print("\n" + "=" * 50)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created At: {format_timestamp(batch.created_at)}")
    print("Request Counts:")
    print(f"  - Total: {batch.request_counts['total']}")
    print(f"  - Completed: {batch.request_counts['completed']}")
    print(f"  - Failed: {batch.request_counts['failed']}")

    # Check various failure states
    failed_states = []
    if batch.failed_at:
        failed_states.append(f"Failed at {format_timestamp(batch.failed_at)}")
    if batch.expired_at:
        failed_states.append(f"Expired at {format_timestamp(batch.expired_at)}")
    if batch.cancelled_at:
        failed_states.append(f"Cancelled at {format_timestamp(batch.cancelled_at)}")

    if failed_states:
        print("Failure Information:")
        for state in failed_states:
            print(f"  - {state}")
    print("=" * 50)


def list_batches():
    """Main function to list and paginate through batches."""
    client = OpenAI()
    last_id = None

    while True:
        try:
            # Get batch list with pagination
            params = {"limit": 100}
            if last_id:
                params["after"] = last_id

            response = client.batches.list(**params)

            # Display batch information
            if not response.data:
                print("\nNo more batches to display.")
                break

            for batch in response.data:
                display_batch_info(batch)

            # Update last_id for pagination
            last_id = response.last_id

            # Ask user if they want to continue
            if not response.has_more:
                print("\nNo more batches available.")
                break

            user_input = input("\nWould you like to see the next page? (yes/no): ").lower()
            if user_input != 'yes' and user_input != 'y':
                break

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break


if __name__ == "__main__":
    print("Starting batch list viewer...")
    list_batches()
