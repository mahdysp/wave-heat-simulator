"""GUI styling and theming configuration"""

# Color palette
COLORS = {
    'bg': '#2b2b2b',
    'fg': '#ffffff',
    'accent': '#4a9eff',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
    'panel': '#3c3c3c',
    'border': '#555555',
    'highlight': '#66bb6a',
    'text_bg': '#1e1e1e',
    'text_fg': '#e0e0e0'
}

def configure_styles(style):
    """Configure ttk styles for dark theme"""
    
    # General styles
    style.configure('TFrame', background=COLORS['bg'])
    style.configure('TLabel', background=COLORS['bg'], foreground=COLORS['fg'])
    style.configure('TButton', 
                   padding=6,
                   relief='flat',
                   background=COLORS['accent'],
                   foreground=COLORS['fg'])
    style.map('TButton', #by-mahdysp
             background=[('active', COLORS['highlight']),
                        ('pressed', COLORS['success'])])
    
    # Notebook styles
    style.configure('TNotebook', background=COLORS['bg'], borderwidth=0)
    style.configure('TNotebook.Tab', 
                   padding=[12, 6],
                   background=COLORS['panel'],
                   foreground=COLORS['fg'])
    style.map('TNotebook.Tab',
             background=[('selected', COLORS['accent'])],
             foreground=[('selected', COLORS['fg'])])
    
    # LabelFrame styles
    style.configure('TLabelframe', 
                   background=COLORS['bg'],
                   bordercolor=COLORS['border'],
                   relief='groove')
    style.configure('TLabelframe.Label', 
                   background=COLORS['bg'],
                   foreground=COLORS['fg'],
                   font=('Helvetica', 10, 'bold'))
    
    # Entry and Spinbox styles
    style.configure('TEntry',
                   fieldbackground=COLORS['text_bg'],
                   foreground=COLORS['text_fg'],
                   insertcolor=COLORS['accent'])
    style.configure('TSpinbox',
                   fieldbackground=COLORS['text_bg'],
                   foreground=COLORS['text_fg'])
    
    # Radiobutton and Checkbutton styles
    style.configure('TRadiobutton',
                   background=COLORS['bg'],
                   foreground=COLORS['fg'])
    style.configure('TCheckbutton',
                   background=COLORS['bg'],
                   foreground=COLORS['fg'])
    
    # Scale styles
    style.configure('Horizontal.TScale',
                   background=COLORS['bg'],
                   troughcolor=COLORS['panel'])
    
    # Combobox styles
    style.configure('TCombobox',
                   fieldbackground=COLORS['text_bg'],
                   foreground=COLORS['text_fg'],
                   selectbackground=COLORS['accent'])
