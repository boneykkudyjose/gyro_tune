import tkinter as tk
from tkinter import filedialog

class TelePrompter:
    def __init__(self, root):
        self.root = root
        self.root.title("TelePrompTer")
        
        self.text_area = tk.Text(root, wrap=tk.WORD, font=("Arial", 20), bg="black", fg="white")
        self.text_area.pack(expand=True, fill=tk.BOTH)
        
        self.scroll_speed = 1  # Default scroll speed
        self.is_scrolling = False
        
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X)
        
        self.speed_scale = tk.Scale(control_frame, from_=1, to=20, orient=tk.HORIZONTAL, label="Speed")
        self.speed_scale.set(self.scroll_speed)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.start_button = tk.Button(control_frame, text="Start", command=self.start_scroll)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_scroll)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.load_button = tk.Button(control_frame, text="Load Text", command=self.load_text)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
    def load_text(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, file.read())
                
    def start_scroll(self):
        self.scroll_speed = self.speed_scale.get()
        self.is_scrolling = True
        self.scroll_text()
        
    def stop_scroll(self):
        self.is_scrolling = False
        
    def scroll_text(self):
        if self.is_scrolling:
            self.text_area.yview_scroll(1, "units")
            self.root.after(1000 // self.scroll_speed, self.scroll_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TelePrompter(root)
    root.mainloop()
