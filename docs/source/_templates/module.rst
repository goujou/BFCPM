{{ name | escape | underline }}

MMMM

.. currentmodule:: {{ fullname }}


.. automodule:: {{ fullname }}

    {% block modules %}
    {% if modules %}
    .. rubric:: Modules
    
    .. autosummary::
       :toctree:
       :recursive:
       :template: module.rst

       {% for item in modules %}
       ~{{ item }}
       {%- endfor %}
    {% endif %}
    {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   {% for item in attributes %}
   .. autodata:: {{ fullname }}::{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :nosignatures:
      :template: class.rst

      {% for class in classes %}
        {{ class }}
      {% endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   {% for item in functions %}
   .. autofunction:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   {% for item in exceptions %}
   .. autoexception:: {{ fullname }}::{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

