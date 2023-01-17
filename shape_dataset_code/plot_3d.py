import numpy as np
import plotly
import plotly.graph_objects as go
import os


def plot_3d(vals,x,y,z, savename = None, display = False, save = True):
    
    xt = np.linspace(1,-1,64)
    yt = np.linspace(1,-1, 64)
    x1,y1 = np.meshgrid(xt,yt)
    z1 = np.ones(x1.shape) * -1
    vals1 = np.zeros(x1.shape)


    color = True
    for i in range(64):
        if i % 8 == 0:
            color = not color
        for j in range(64):

            if j % 8 == 0:
                color = not color
            vals1[i,j] = int(color)


    fig = go.Figure(data=go.Surface(
            x = x1,
            y = y1,
            z = z1,
            surfacecolor = vals1,
            colorscale = [[0,"white"], [1,"black"]],
            showscale = False,

        lighting=dict(ambient=0.5,
                 diffuse=1,
                 fresnel=2,        
                 specular=0.5,
                 roughness=0.5),
       lightposition = dict(x=0,
                    y=0,
                    z=2)

            ), layout = go.Layout(autosize =False, width = 350, height = 350))


    fig.add_trace(go.Surface(
            x = z1*-1,
            y = y1,
            z = x1,
            surfacecolor = vals1,
            colorscale = [[0,"white"], [1,"black"]],
            showscale = False,

        lighting=dict(ambient=0.5,
                 diffuse=1,
                 fresnel=2,        
                 specular=0.5,
                 roughness=0.5),
       lightposition = dict(x=-2,
                    y=0,
                    z=2)

            ))


    fig.add_trace(go.Surface(
            x = x1,
            y = z1,
            z = y1,
            surfacecolor = vals1,
            colorscale = [[0,"white"], [1,"black"]],
            showscale = False,

        lighting=dict(ambient=0.5,
                 diffuse=1,
                 fresnel=2,        
                 specular=0.5,
                 roughness=0.5),
       lightposition = dict(x=-2,
                    y=0,
                    z=2)

            ))

    fig.add_trace(go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=vals.flatten(),
        #value = convolved_vals.flatten(),
        cmax = 1,
        cmin = -1,
        isomin=.1,
        isomax=1,
        showscale = False,
        #colorscale = greens,
        opacity=1, # needs to be small to see through all surfaces
        surface_count=21,
        lighting=dict(ambient=0.5,
                 diffuse=1,
                 fresnel=2,        
                 specular=0.5,
                 roughness=0.5),
       lightposition = dict(x=0,
                    y=0,
                    z=2)

    )) # needs to be a large number for good volume rendering

    fig.update_traces(lighting=dict(ambient=0.1,
                 diffuse=1,
                 fresnel=2,        
                 specular=0.5,
                 roughness=0.5),
       lightposition = dict(x=-2,
                    y=2,
                    z=2))
    
    
    fig['layout'].update(scene = dict(#aspectmode = "manual",
        aspectratio = dict(x = 1, y = 1, z = 1),
        xaxis = dict(showgrid = False, zeroline = False, showline = False, autorange = True, ticks = '', showticklabels = False, title = '', showbackground = False),
        yaxis = dict(showgrid = False, zeroline = False, showline = False, autorange = True, ticks = '', showticklabels = False, title = '', showbackground = False),
        zaxis = dict(showgrid = False, zeroline = False, showline = False,  autorange = True, ticks = '', showticklabels = False, title = '', showbackground = False)),
        scene_camera = dict(eye=dict(x=0, y=1.5, z = 1.5)),
        #this scene camera looks at the americas from the equator
        #z = 0 turns out to be level with the equator 
        #(maybe since z range -1.3ish to 1.3ish)
        margin = dict(l=0,r=0,b=0,t=0,pad=2),
        )

    if display:
        fig.show()
    if save:
        fig.write_image(savename)
    #return fig
    del fig


if __name__ == "__main__":

    import sys
    import os
    import time
    import subprocess

    plotly.io.orca.config.executable = "C:/Users/tdelmatt/Anaconda3/envs/plotlyenv/orca_app/orca.exe"
    output_dir = sys.argv[1]
    assert(os.path.isdir(output_dir))
    vals = np.load(output_dir + "/output.npy")
    
    assert(vals.shape[0] == 64 and vals.shape[1] == 64 and vals.shape[2] == 64)
    vals[(vals > .7)] = 1
    vals[(vals < .3)] = 0
    
    x,y,z = np.meshgrid(np.linspace(-1,1,64), np.linspace(-1,1,64), np.linspace(-1,1,64))
    savename = output_dir + '/3d_predict.png'
    plot_3d(vals, x, y, z, savename=savename, display=False, save=True)
    
    
        