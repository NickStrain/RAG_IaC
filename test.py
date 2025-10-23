import json
from typing import List, Dict
import random
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
import os

console = Console()


class TerraformDocsSampleViewer:
    def __init__(self, json_file: str = 'terraform_aws_docs.json'):
        """Initialize the sample viewer"""
        self.json_file = json_file
        self.documents = []
        self.load_documents()
    
    def load_documents(self):
        """Load documents from JSON file"""
        if not os.path.exists(self.json_file):
            console.print(f"[red]❌ Error: {self.json_file} not found![/red]")
            console.print("[yellow]Please run the scraper script first to generate the data.[/yellow]")
            return
        
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            console.print(f"[green]✓ Loaded {len(self.documents)} documents from {self.json_file}[/green]\n")
        except Exception as e:
            console.print(f"[red]Error loading documents: {e}[/red]")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the scraped documents"""
        if not self.documents:
            return {}
        
        stats = {
            'total': len(self.documents),
            'resources': sum(1 for doc in self.documents if doc['type'] == 'resource'),
            'data_sources': sum(1 for doc in self.documents if doc['type'] == 'data_source'),
            'with_examples': sum(1 for doc in self.documents if doc.get('example', '').strip()),
            'with_arguments': sum(1 for doc in self.documents if doc.get('arguments', '').strip()),
            'avg_description_length': sum(len(doc.get('description', '')) for doc in self.documents) / len(self.documents),
            'avg_full_text_length': sum(len(doc.get('full_text', '')) for doc in self.documents) / len(self.documents)
        }
        
        return stats
    
    def display_statistics(self):
        """Display statistics in a nice table"""
        stats = self.get_statistics()
        
        if not stats:
            console.print("[red]No documents loaded![/red]")
            return
        
        table = Table(title="📊 Scraped Documents Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Documents", str(stats['total']))
        table.add_row("Resources", str(stats['resources']))
        table.add_row("Data Sources", str(stats['data_sources']))
        table.add_row("Documents with Examples", str(stats['with_examples']))
        table.add_row("Documents with Arguments", str(stats['with_arguments']))
        table.add_row("Avg Description Length", f"{stats['avg_description_length']:.0f} chars")
        table.add_row("Avg Full Text Length", f"{stats['avg_full_text_length']:.0f} chars")
        
        console.print(table)
        console.print()
    
    def display_document(self, doc: Dict, show_full_text: bool = False):
        """Display a single document in a formatted way"""
        # Header
        title = f"📄 {doc['name']} ({doc['type']})"
        console.print(Panel(title, style="bold blue"))
        
        # URL
        console.print(f"[bold]URL:[/bold] [link]{doc['url']}[/link]\n")
        
        # Description
        if doc.get('description'):
            console.print("[bold]Description:[/bold]")
            console.print(Panel(doc['description'], border_style="green"))
        
        # Arguments
        if doc.get('arguments'):
            console.print("\n[bold]Arguments:[/bold]")
            args_text = doc['arguments'][:500] + "..." if len(doc['arguments']) > 500 else doc['arguments']
            console.print(Panel(args_text, border_style="yellow"))
        
        # Attributes
        if doc.get('attributes'):
            console.print("\n[bold]Attributes:[/bold]")
            attrs_text = doc['attributes'][:500] + "..." if len(doc['attributes']) > 500 else doc['attributes']
            console.print(Panel(attrs_text, border_style="cyan"))
        
        # Example
        if doc.get('example'):
            console.print("\n[bold]Example Code:[/bold]")
            example_text = doc['example'][:800] + "\n..." if len(doc['example']) > 800 else doc['example']
            console.print(Panel(example_text, border_style="magenta", title="Terraform HCL"))
        
        # Full text (optional)
        if show_full_text and doc.get('full_text'):
            console.print("\n[bold]Full Text Preview:[/bold]")
            full_text = doc['full_text'][:1000] + "\n..." if len(doc['full_text']) > 1000 else doc['full_text']
            console.print(Panel(full_text, border_style="red"))
        
        console.print("\n" + "="*80 + "\n")
    
    def show_random_samples(self, count: int = 5, show_full_text: bool = False):
        """Show random sample documents"""
        if not self.documents:
            console.print("[red]No documents loaded![/red]")
            return
        
        sample_size = min(count, len(self.documents))
        samples = random.sample(self.documents, sample_size)
        
        console.print(f"[bold green]Showing {sample_size} random samples:[/bold green]\n")
        
        for i, doc in enumerate(samples, 1):
            console.print(f"[bold yellow]Sample {i}/{sample_size}[/bold yellow]")
            self.display_document(doc, show_full_text)
    
    def show_by_type(self, doc_type: str, count: int = 3, show_full_text: bool = False):
        """Show samples of a specific type"""
        filtered = [doc for doc in self.documents if doc['type'] == doc_type]
        
        if not filtered:
            console.print(f"[red]No documents of type '{doc_type}' found![/red]")
            return
        
        sample_size = min(count, len(filtered))
        samples = random.sample(filtered, sample_size)
        
        console.print(f"[bold green]Showing {sample_size} {doc_type} samples:[/bold green]\n")
        
        for i, doc in enumerate(samples, 1):
            console.print(f"[bold yellow]Sample {i}/{sample_size}[/bold yellow]")
            self.display_document(doc, show_full_text)
    
    def search_by_name(self, search_term: str, show_full_text: bool = False):
        """Search for documents by name"""
        matches = [doc for doc in self.documents if search_term.lower() in doc['name'].lower()]
        
        if not matches:
            console.print(f"[red]No documents found matching '{search_term}'[/red]")
            return
        
        console.print(f"[bold green]Found {len(matches)} documents matching '{search_term}':[/bold green]\n")
        
        for i, doc in enumerate(matches[:5], 1):  # Show first 5 matches
            console.print(f"[bold yellow]Match {i}[/bold yellow]")
            self.display_document(doc, show_full_text)
    
    def list_all_names(self, doc_type: str = None):
        """List all document names"""
        if doc_type:
            filtered = [doc for doc in self.documents if doc['type'] == doc_type]
            title = f"All {doc_type} Names"
        else:
            filtered = self.documents
            title = "All Document Names"
        
        table = Table(title=title, box=box.SIMPLE)
        table.add_column("#", style="cyan", width=6)
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        
        for i, doc in enumerate(filtered, 1):
            table.add_row(str(i), doc['name'], doc['type'])
            if i >= 50:  # Limit to 50 for readability
                console.print(table)
                console.print(f"\n[yellow]... and {len(filtered) - 50} more documents[/yellow]\n")
                return
        
        console.print(table)
        console.print()
    
    def export_sample_to_file(self, count: int = 10, output_file: str = 'sample_docs.txt'):
        """Export sample documents to a text file"""
        if not self.documents:
            console.print("[red]No documents loaded![/red]")
            return
        
        sample_size = min(count, len(self.documents))
        samples = random.sample(self.documents, sample_size)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"TERRAFORM AWS PROVIDER DOCUMENTATION SAMPLES\n")
            f.write(f"Total samples: {sample_size}\n")
            f.write("="*80 + "\n\n")
            
            for i, doc in enumerate(samples, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"SAMPLE {i}/{sample_size}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Name: {doc['name']}\n")
                f.write(f"Type: {doc['type']}\n")
                f.write(f"URL: {doc['url']}\n\n")
                f.write(f"Description:\n{doc.get('description', 'N/A')}\n\n")
                
                if doc.get('arguments'):
                    f.write(f"Arguments:\n{doc['arguments'][:500]}...\n\n")
                
                if doc.get('example'):
                    f.write(f"Example:\n{doc['example'][:800]}...\n\n")
                
                f.write(f"Full Text Preview:\n{doc.get('full_text', 'N/A')[:1000]}...\n\n")
        
        console.print(f"[green]✓ Exported {sample_size} samples to {output_file}[/green]\n")


def main():
    """Main function with interactive menu"""
    viewer = TerraformDocsSampleViewer()
    
    if not viewer.documents:
        return
    
    while True:
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]       TERRAFORM DOCUMENTATION SAMPLE VIEWER[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
        
        console.print("[bold]Options:[/bold]")
        console.print("  [cyan]1[/cyan] - Show statistics")
        console.print("  [cyan]2[/cyan] - Show random samples (5)")
        console.print("  [cyan]3[/cyan] - Show resource samples")
        console.print("  [cyan]4[/cyan] - Show data source samples")
        console.print("  [cyan]5[/cyan] - Search by name")
        console.print("  [cyan]6[/cyan] - List all names")
        console.print("  [cyan]7[/cyan] - Export samples to file")
        console.print("  [cyan]8[/cyan] - Show random samples with full text")
        console.print("  [cyan]9[/cyan] - Quick view (3 random docs)")
        console.print("  [cyan]0[/cyan] - Exit")
        
        choice = console.input("\n[bold yellow]Select an option: [/bold yellow]")
        
        console.print()
        
        if choice == '1':
            viewer.display_statistics()
        
        elif choice == '2':
            viewer.show_random_samples(count=5, show_full_text=False)
        
        elif choice == '3':
            viewer.show_by_type('resource', count=3, show_full_text=False)
        
        elif choice == '4':
            viewer.show_by_type('data_source', count=3, show_full_text=False)
        
        elif choice == '5':
            search = console.input("[bold]Enter search term (e.g., 's3', 'ec2', 'lambda'): [/bold]")
            viewer.search_by_name(search, show_full_text=False)
        
        elif choice == '6':
            type_choice = console.input("[bold]Filter by type? (resource/data_source/all): [/bold]")
            if type_choice in ['resource', 'data_source']:
                viewer.list_all_names(type_choice)
            else:
                viewer.list_all_names()
        
        elif choice == '7':
            count = console.input("[bold]How many samples to export? (default 10): [/bold]")
            count = int(count) if count.isdigit() else 10
            viewer.export_sample_to_file(count=count)
        
        elif choice == '8':
            viewer.show_random_samples(count=3, show_full_text=True)
        
        elif choice == '9':
            viewer.show_random_samples(count=3, show_full_text=False)
        
        elif choice == '0':
            console.print("[bold green]Goodbye! 👋[/bold green]")
            break
        
        else:
            console.print("[red]Invalid option. Please try again.[/red]")


if __name__ == "__main__":
    main()