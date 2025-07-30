from django.db import models

class Interaction(models.Model):
    ACTION_CHOICES = [
        ('code_generation', 'Code Generation'),
        ('debugging',       'Debugging'),
        ('git_operation',   'Git Operation'),
        ('analyze_file',    'Analyze File'),
        ('generate_ideas',  'Generate Ideas'),
        ('general_ai',      'General AI'),
    ]

    prompt      = models.TextField()
    result      = models.TextField()
    action_type = models.CharField(max_length=30, choices=ACTION_CHOICES)
    created_at  = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.get_action_type_display()}] {self.prompt[:50]!r}"


# Create your models here.
